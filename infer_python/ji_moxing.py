import json
import argparse
import os

os.system('pip uninstall tools')
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

ROOT = '/project/train/src_repo/'  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# DET_ROOT = Path(os.path.relpath(DET_ROOT, Path.cwd()))  # relative

DET_ROOT = '/project/train/src_repo/yolov5_vehicle_plate_det/'  # YOLOv5 root directory
if str(DET_ROOT) not in sys.path:
    sys.path.append(str(DET_ROOT))  # add ROOT to PATH

sys.path.append('/project/train/src_repo/PaddleOCR/tools/')

OCR_ROOT = '/project/train/src_repo/PaddleOCR/'  # YOLOv5 root directory
if str(OCR_ROOT) not in sys.path:
    sys.path.append(str(OCR_ROOT))  # add ROOT to PATH

# YOLO Det
from yolov5_det.models.common import DetectMultiBackend
from yolov5_det.utils.dataloaders import ji_LoadImages
from yolov5_det.utils.general import (cv2, check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5_det.utils.torch_utils import select_device
# paddle ocr
import PaddleOCR.tools
from PaddleOCR.tools import plate_infer_rec

# 参数
# /project/train/models/exp3/weights/best.pt
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='/project/train/models/exp3/weights/last.pt',
                    help='exp3 model path(s)')
parser.add_argument('--data', type=str, default='/project/train/src_repo/yolov5_vehicle_plate_det/data/vehicle.yaml',
                    help='(optional) dataset.yaml path')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
parser.add_argument('--conf-thres', type=float, default=0.15, help='confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.30, help='NMS IoU threshold')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

parser.add_argument('--none_w', type=int, default=33, help='use OpenCV DNN for ONNX inference')
parser.add_argument('--none_h', type=int, default=17, help='use OpenCV DNN for ONNX inference')
parser.add_argument('--none_wh', type=int, default=1000, help='576')

opt = parser.parse_args()
opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand


cls_names = ['truck', 'van', 'car', 'slagcar', 'bus', 'fire_truck',
             'police_car', 'ambulance', 'SUV', 'microbus', 'unknown_vehicle',
             'plate', 'double_plate']  # class names
vehicle_color = ['white', 'siliver', 'grey', 'black', 'red', 'blue', 'yellow', 'green', 'brown', 'others']


import itertools
pred_frames_list = []

@torch.no_grad()
def init():
    '''Initialize model
    Returns: model
    '''
    opt.half = True
    # Load model
    device = select_device(opt.device)
    det_model = DetectMultiBackend(opt.weights, device=device, dnn=opt.dnn, data=opt.data, fp16=opt.half)
    ocr_model = plate_infer_rec.ocr_init_model()

    return [det_model, ocr_model]


def process_image(handle=None, input_image=None, args=None, **kwargs):
    '''Do inference to analysis input_image and get output
    Attributes:
    handle: algorithm handle returned by init()
    input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
    Returns: process result
    '''

    def isChinese(word):
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False

    det_model = handle[0]
    ocr_model = handle[1]


    # det model
    stride, names, pt = det_model.stride, det_model.names, det_model.pt
    imgsz = check_img_size(opt.imgsz, s=stride)  # check image size
    im, im0s = ji_LoadImages(input_image, imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    # Run inference
    det_model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    device = select_device(opt.device)
    im = torch.from_numpy(im).to(device)
    im = im.half() if det_model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # det Inference
    pred = det_model(im, augment=opt.augment, visualize=False)
    # NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)


    # Process predictions
    fake_result = {}
    fake_result = {'objects': []}

    target_count = 0
    is_alert = False
    target_info = []

    for i, det in enumerate(pred):  # per image
        # det xywh + conf + classes + colors
        im0 = im0s.copy()

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls, clr in reversed(det):

                name = cls_names[int(cls)]
                color = vehicle_color[int(clr)]

                # plate ocr
                # 如果是车牌，进行车牌识别
                if int(cls) == 11 or int(cls) == 12:
                    # 计算像素，像素处于某一范围内直接设置为“NONE”
                    plate_w = xyxy[2] - xyxy[0]
                    plate_h = xyxy[3] - xyxy[1]
                    plate_wh = plate_w * plate_h
                    # NONE 处理
                    if (plate_w < opt.none_w) or (plate_h < opt.none_h) or (plate_wh < opt.none_wh):
                        plate_content = 'NONE'
                    else:
                        plate_img = input_image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        ocr_content = plate_infer_rec.ocr_predict(ocr_model, plate_img)
                        # ~ _ 后处理
                        plate_content = ocr_content[0]

                        if len(plate_content) < 7 and (not isChinese(plate_content)):
                            for ii in range(7 - len(plate_content)):
                                if ii != 0:
                                    plate_content = '_' + plate_content

                            plate_content = '~' + plate_content

                        if len(plate_content) < 7 and isChinese(plate_content):
                            for ii in range(7 - len(plate_content)):
                                plate_content = plate_content + '_'


                    fake_result['objects'].append(
                        {
                            "points": [
                                [int(xyxy[0]), int(xyxy[1])],
                                [int(xyxy[2]), int(xyxy[1])],
                                [int(xyxy[2]), int(xyxy[3])],
                                [int(xyxy[0]), int(xyxy[3])]],
                            "name": str(name),
                            "confidence": float(conf),
                            "ocr": str(plate_content),
                            "color": str(color)
                        })

                else:

                    fake_result['objects'].append(
                        {
                            "xmin": int(xyxy[0]),
                            "ymin": int(xyxy[1]),
                            "xmax": int(xyxy[2]),
                            "ymax": int(xyxy[3]),
                            "confidence": float(conf),
                            "name": str(name),
                            "color": str(color)
                        })

                    # Process image here

    return json.dumps(fake_result, indent=4)


if __name__ == '__main__':
    # Test API
    #     img =cv2.imread('/home/data/1112/ZDSvehicle20220613_V8_train_street_27_000096.jpg')
    #     predictor = init()
    #     import time
    #     s = time.time()
    #     fake_result = process_image(predictor, img)
    #     e = time.time()
    #     print(fake_result)
    #     print((e-s))

    # test ocr
    # /project/train/src_repo/PaddleOCR/plate_test.jpg  /home/data/1112/ZDSvehicle20220613_V8_train_street_27_000096.jpg
    img = cv2.imread('/project/train/src_repo/PaddleOCR/plate_test.jpg')
    predictor = init()
    import time

    s = time.time()
    fake_result = process_image(predictor, img)
    e = time.time()
    print(fake_result)
    print((e - s))