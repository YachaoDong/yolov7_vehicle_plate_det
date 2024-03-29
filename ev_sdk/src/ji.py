import json
import argparse
import os
# os.system('cp -a /usr/lib/tensorrt/* /usr/local/lib/')
# os.system('make pyinstall -j8 -C /project/train/src_repo/tensorRT_Pro/')

import sys
from pathlib import Path
import torch
import itertools
import cv2
ROOT = '/project/train/src_repo/'  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# DET_ROOT = Path(os.path.relpath(DET_ROOT, Path.cwd()))  # relative

DET_ROOT = '/project/train/src_repo/yolov7_vehicle_plate_det/'  # YOLOv5 root directory
if str(DET_ROOT) not in sys.path:
    sys.path.append(str(DET_ROOT))  # add ROOT to PATH

sys.path.append('/project/train/src_repo/PaddleOCR/tools/')

OCR_ROOT = '/project/train/src_repo/PaddleOCR/'  # YOLOv5 root directory
if str(OCR_ROOT) not in sys.path:
    sys.path.append(str(OCR_ROOT))  # add ROOT to PATH


# paddle ocr
import PaddleOCR.tools
from PaddleOCR.tools import plate_infer_rec as ocr_pred
# from PaddleOCR.tools.infer import plate_predict_rec as ocr_pred


# tensorrt infer
PYTRT_ROOT = '/project/train/src_repo/tensorRT_Pro-main/example-python/'
if str(PYTRT_ROOT) not in sys.path:
    sys.path.append(str(PYTRT_ROOT))
import pytrt as tp



# 参数
# /project/train/models/exp3/weights/best.pt
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='/project/train/models/exp3/weights/last.pt',
                    help='exp3 model path(s)')

parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# ocr NONE阈值
parser.add_argument('--none_w', type=int, default=33, help='use OpenCV DNN for ONNX inference')
parser.add_argument('--none_h', type=int, default=17, help='use OpenCV DNN for ONNX inference')
parser.add_argument('--none_wh', type=int, default=1000, help='576')

parser.add_argument('--alert_obj', type=str, default='vehicle', help='use plate/vehicle info alert')
# /project/train/models/exp/weights/deploy_yolov7_last.trt
# /usr/local/ev_sdk/model1/35403/last.trt
parser.add_argument('--engine_file', type=str, default='/project/train/models/exp/weights/last_165.trt', help='use plate/vehicle info alert')
parser.add_argument('--onnx_file', type=str, default='/project/train/models/exp/weights/deploy_yolov7_last.onnx', help='use plate/vehicle info alert')

opt = parser.parse_args()

cls_names = ['truck', 'van', 'car', 'slagcar', 'bus', 'fire_truck',
             'police_car', 'ambulance', 'SUV', 'microbus', 'unknown_vehicle',
             'plate', 'double_plate']  # class names
vehicle_color = ['white', 'siliver', 'grey', 'black', 'red', 'blue', 'yellow', 'green', 'brown', 'others']




def init():
    '''Initialize model
    Returns: model
    '''
    print("use init")
    ocr_model = ocr_pred.ocr_init_model()
    
    # tensorrt
    # tp.compile_onnx_to_file(1, opt.onnx_file, opt.engine_file)
    trt_det_model = tp.Yolo(opt.engine_file, type=tp.YoloType.V7, confidence_threshold=opt.conf_thres, nms_threshold=opt.iou_thres, nms_method=tp.NMSMethod.FastGPU)

    return [' ', ocr_model, trt_det_model]


def process_image(handle=None, input_image=None, args=None, **kwargs):
    '''Do inference to analysis input_image and get output
    Attributes:
    handle: algorithm handle returned by init()
    input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
    Returns: process result
    '''
    print('handle[2]:', handle[2])
    def isChinese(word):
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False


    ocr_model = handle[1]
    # trt det model
    trt_det_model = handle[2]

    
    fake_result = {'objects': []}
    
    # detect_info = {'vehicle': [  [('vehicle_name', 'vehicl_color'), (x_min,y_min,x_max,y_max)], ...,[]  ],
    #                  'plate': [  [('plate_ocr',  'plate_color'),  (x_min,y_min,x_max,y_max)], ...,[]  ]}
    detect_info = {'vehicle': [], 'plate': []}  
    
    # inference
    # det = trt_det_model.commit(input_image).get()
    
    # width = input_image.shape[1],
    # height = input_image.shape[0],
    
    gpu_image = torch.from_numpy(input_image).to(0)
    det = trt_det_model.commit_gpu(
        pimage = gpu_image.data_ptr(),
        width = input_image.shape[1],
        height = input_image.shape[2],
        device_id = 0,
        imtype = tp.ImageType.GPUBGR,
        stream = torch.cuda.current_stream().cuda_stream).get()
    print('det:', det)
    # 对det 按照 顺序排序
    # det:[<Box: left=797.24, top=715.03, right=924.00, bottom=754.33, class_label=11, color_label=5, confidence=0.72818>, ]
    if len(det):
        sort_det = []
        # Write results
        for box in det:
            x_min, y_min, x_max, y_max = map(int, [box.left, box.top, box.right, box.bottom])
            conf = float(box.confidence)
            cls = int(box.class_label)
            clr = int(box.color_label)
            sort_det.append([x_min, y_min, x_max, y_max, conf, cls, clr])
        
        sort_det = sorted(sort_det, key=lambda x:x[4], reverse=True)
        
        for box in sort_det:
            x_min, y_min, x_max, y_max = box[:4]
            conf = box[4]
            cls = box[5]
            clr = box[6]
            
            name = str(cls_names[cls])
            color = str(vehicle_color[clr])

            # ------------ get plate ocr -----------------------------------
            # 如果是车牌，进行车牌识别
            if int(cls) == 11 or int(cls) == 12:
                # 计算像素，像素处于某一范围内直接设置为“NONE”
                plate_w = x_max - x_min
                plate_h = y_max - y_min
                plate_wh = plate_w * plate_h
                # NONE 处理
                if (plate_w < opt.none_w) or (plate_h < opt.none_h) or (plate_wh < opt.none_wh):
                # if True:
                    plate_content = 'NONE'
                else:
                    plate_img = input_image[y_min:y_max, x_min:x_max]
                    
                    
                    # plate_img[plate_img<0] = 0
                    try:
                        ocr_content = ocr_pred.ocr_predict(ocr_model, plate_img)
                    except:
                        print(f'y_min:{y_min},  y_max:{y_max},  x_min:{x_min}, x_max:{x_max}')
                        print('plate_img:', plate_img)
                    # ocr_content = ocr_model([plate_img])[0][0]
                    plate_content = ocr_content[0]

                    # ~ _ 后处理
                    if len(plate_content) < 7 and (not isChinese(plate_content)):
                        for ii in range(7 - len(plate_content)):
                            if ii != 0:
                                plate_content = '_' + plate_content
                        plate_content = '~' + plate_content
                    
                    if len(plate_content) < 7 and isChinese(plate_content):
                        for ii in range(7 - len(plate_content)):
                            plate_content = plate_content + '_'

                # ------------ get plate ocr ----------------------------------------
                fake_result['objects'].append(
                        {
                            "points": [
                                [int(x_min), int(y_min)],
                                [int(x_max), int(y_min)],
                                [int(x_max), int(y_max)],
                                [int(x_min), int(y_max)]],
                            "name": str(name),
                            "confidence": float(conf),
                            "ocr": str(plate_content),
                            "color": str(color)
                        })
                
            else:
                fake_result['objects'].append(
                        {
                            "xmin": int(x_min),
                            "ymin": int(y_min),
                            "xmax": int(x_max),
                            "ymax": int(y_max),
                            "confidence": float(conf),
                            "name": str(name),
                            "color": str(color)
                        })


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
    img = cv2.imread('/project/train/src_repo/1.jpg')
    predictor = init()
    import time

    s = time.time()
    fake_result = process_image(predictor, img)
    e = time.time()
    print(fake_result)
    print((e - s))