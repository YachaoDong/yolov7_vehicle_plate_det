#run.sh
# bash /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/run.sh

# 如果存在要保存的文件，提前删除文件夹
rm  -r /home/data/vehicle_data/*

 

#创建数据集相关文件夹
# mkdir  -p /project/.config/Ultralytics/
mkdir  -p /home/data/vehicle_data/labels

# 生成all imgs abs path txt
find /home/data/1*/ -name "*.jpg" | xargs -i ls {}  > /home/data/vehicle_data/all_det_imgs.txt

# find /home/data/7*/ -name "*.jpg" | xargs -i ls {}  > /home/data/vehicle_data/all_ocr_imgs.txt




# xml转txt labels
python /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/xml2labels.py



#执行YOLOV5训练脚本
# pip install -r /project/train/src_repo/yolov5_vehicle_plate_det/requirements.txt 

# cp /project/train/src_repo/yolov5_vehicle_plate_det/yolov5_det/Arial.ttf  /project/.config/Ultralytics/

# python /project/train/src_repo/yolov5_vehicle_plate_det/train.py   --batch-size 32 --weights /project/train/models/exp/weights/best.pt  --epochs 180 --workers 4 --multi-scale


# 开发环境训练demo 调试
python /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/train.py   --batch-size 4 --weights /project/train/src_repo/yolov7_vehicle_plate_det/yolov7_training.pt --epochs 1 --workers 4

# 正式训练
# python /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/train.py   --batch-size 16 --weights /project/train/models/exp/weights/last.pt  --resume  /project/train/models/exp/weights/last.pt --epochs 200 --workers 8 --image-weights

# python /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/train.py   --batch-size 16 --weights /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/yolov7_training.pt  --data /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/data/vehicle.yaml  --hyp /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/data/hyp.scratch.vehicle_custom.yaml --epochs 100 --workers 4 --multi-scale


# 测试
# python /project/train/src_repo/yolov5_vehicle_plate_det/yolov5_det/val.py --data /project/train/src_repo/yolov5_vehicle_plate_det/yolov5_det/data/vehicle.yaml --weights /project/train/models/exp2/weights/last.pt



