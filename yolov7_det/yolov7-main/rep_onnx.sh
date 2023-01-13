# bash /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/rep_onnx.sh

cd /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/

# 训练模型转推理模型
python YOLOv7_reparameterization.py --training_weights /project/train/models/exp/weights/last.pt --deploy_yaml /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/cfg/deploy/vehicle_yolov7_deploy.yaml --deploy_weights /project/train/models/exp/weights/deploy_yolov7_last.pt

# 推理模型转onnx
python export_trt.py --dynamic --grid --weight=/project/train/models/exp/weights/deploy_yolov7_last.pt



# # 训练模型转推理模型
# python YOLOv7_reparameterization.py --training_weights /project/train/models/exp/weights/best.pt --deploy_yaml /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/cfg/deploy/vehicle_yolov7_deploy.yaml --deploy_weights /project/train/models/exp/weights/deploy_yolov7_best.pt

# # 推理模型转onnx
# python export_trt.py --dynamic --grid --weight=/project/train/models/exp/weights/deploy_yolov7_best.pt



