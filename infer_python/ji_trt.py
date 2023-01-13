import json
import argparse
import os

import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import time
import glob
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
from yolov5_vehicle_plate_det.models.common import DetectMultiBackend
from yolov5_vehicle_plate_det.utils.dataloaders import ji_LoadImages
from yolov5_vehicle_plate_det.utils.general import (cv2, check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5_vehicle_plate_det.utils.torch_utils import select_device
# paddle ocr
# import PaddleOCR.tools
# from PaddleOCR.tools import plate_infer_rec as ocr_pred
# from PaddleOCR.tools.infer import plate_predict_rec as ocr_pred


# tensorrt infer
PYTRT_ROOT = '/project/train/src_repo/tensorRT_Pro/example-python/'
if str(PYTRT_ROOT) not in sys.path:
    sys.path.append(str(PYTRT_ROOT))

import pytrt as tp

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

parser.add_argument('--alert_obj', type=str, default='vehicle', help='use plate/vehicle info alert')

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
    stride, names, pt = det_model.stride, det_model.names, det_model.pt
    imgsz = check_img_size(opt.imgsz, s=stride)  # check image size
    bs = 1  # batch_size
    det_model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    ocr_model = ''
    
    # tensorrt
    engine_file = "/project/train/models/my_dett.FP32.trtmodel"
    # tp.compile_onnx_to_file(1, '/project/train/models/exp3/weights/last.onnx', engine_file)
    # engine_file = '/project/train/models/vehicle_yolov5.FP32.trtmodel'
    # try:
    #     if not os.path.exists(engine_file):
    #         tp.compile_onnx_to_file(1, '/project/train/models/exp3/weights/last.onnx', engine_file)
    # except:
    #     engine_file = '/project/train/models/vehicle_yolov5.FP32.trtmodel'
    # tp.compile_onnx_to_file(1, '/project/train/models/exp3/weights/last.onnx', engine_file)
    
    tp.compile_onnx_to_file(1, '/project/train/models/exp3/weights/last.onnx', engine_file)
    trt_det_model = tp.Yolo(engine_file, type=tp.YoloType.V5, confidence_threshold=opt.conf_thres, nms_threshold=opt.iou_thres)
    

    return [(det_model, stride, names, pt, imgsz, device), ocr_model, trt_det_model]


def process_image(handle=None, input_image=None, args=None, **kwargs):

    det_model, stride, names, pt, imgsz, device = handle[0]
    ocr_model = handle[1]
    trt_det_model = handle[2]
    
    t1 = time.time()
    trt_bboxes = trt_det_model.commit(input_image).get()
    trt_time = time.time()-t1
    print('***trt_bboxes***:', trt_bboxes)
    print('***trt cost time***:', trt_time)

    

    # Run inference
    t2 = time.time()
    im, im0s = ji_LoadImages(input_image, imgsz, stride=stride, auto=pt)
    im = torch.from_numpy(im).to(device)
    im = im.half() if det_model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0

    trt_bboxes = trt_det_model.commit_gpu(
        pimage = im.data_ptr(),
        width = im0s.shape[1],
        height = im0s.shape[0],
        device_id = 0,
        imtype = tp.ImageType.GPUBGR,
        stream = torch.cuda.current_stream().cuda_stream).get()
     if len(trt_bboxes):
            sort_det = []
            # Write results
            for box in det:
                x_min, y_min, x_max, y_max = map(int, [box.left, box.top, box.right, box.bottom])
                conf = float(box.confidence)
                cls = int(box.class_label)
                clr = int(box.color_label)
                sort_det.append([x_min, y_min, x_max, y_max, conf, cls, clr])
            sort_det = sorted(sort_det, key=lambda x:x[4], reverse=True)
            print('***trt_sort_det***:', sort_det)   
    trt_time = time.time()-t1
    print('***trt_bboxes222***:', trt_bboxes)
    print('***trt cost time***:', trt_time)
    
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    # det Inference
    t1 = time.time()

    pred = det_model(im, augment=opt.augment, visualize=False)
    # NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
    det = pred[0]
    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
    print('***det_2***:', det)


    


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
    # img = cv2.imread('/project/train/src_repo/PaddleOCR/plate_test.jpg')
    predictor = init()

    img_path_list = glob.glob('/home/data/1111/*.jpg')
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        process_image(predictor, img)


    '''
    py::class_<YoloInfer>(m, "Yolo")
    .def(py::init<string, YoloGPUPtr::Type, int, float, float, YoloGPUPtr::NMSMethod, int>(), 
        py::arg("engine"), 
        py::arg("type")                 = YoloGPUPtr::Type::V5, 
        py::arg("device_id")            = 0, 
        py::arg("confidence_threshold") = 0.4f,
        py::arg("nms_threshold") = 0.5f,
        py::arg("nms_method")    = YoloGPUPtr::NMSMethod::FastGPU,
        py::arg("max_objects")   = 1024
    )
    .def_property_readonly("valid", &YoloInfer::valid, "Infer is valid")
    .def("commit", &YoloInfer::commit, py::arg("image"))
    .def("commit_gpu", &YoloInfer::commit_gpu, 
        py::arg("pimage"), py::arg("width"), py::arg("height"), py::arg("device_id"), py::arg("imtype"), py::arg("stream")
    );
    
	m.def(
		"compileTRT", compileTRT,
		py::arg("max_batch_size"),
		py::arg("source"),
		py::arg("output"),
		py::arg("mode")                         = TRT::Mode::FP32,
		py::arg("inputs_dims")                  = py::array_t<int>(),
		py::arg("device_id")                    = 0,
		py::arg("int8_norm")                    = CUDAKernel::Norm::None(),
		py::arg("int8_preprocess_const_value")  = 114,
		py::arg("int8_image_directory")         = ".",
		py::arg("int8_entropy_calibrator_file") = "",
		py::arg("max_workspace_size")           = 1ul << 30
	);
    
    '''