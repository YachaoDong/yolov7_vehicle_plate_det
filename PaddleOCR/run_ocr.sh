#run.sh

# 如果存在要保存的文件，提前删除文件夹
rm  -r /home/data/ocr_data/*

 

#创建数据集相关文件夹
# mkdir  -p /project/.config/Ultralytics/
mkdir  -p /home/data/ocr_data/

# 生成all imgs abs path txt
find /home/data/1*/ -name "*.jpg" | xargs -i ls {}  > /home/data/ocr_data/all_det_imgs.txt

# find /home/data/1*/ -name "*.jpg" | xargs -i ls {}  > /home/data/vehicle_data/all_ocr_imgs.txt

# pip install -r /project/train/src_repo/PaddleOCR/requirements.txt

python /project/train/src_repo/PaddleOCR/ocr_process.py
# python /project/train/src_repo/PaddleOCR/tools/train.py -c /project/train/src_repo/PaddleOCR/configs/rec/plate_rec_mv3_none_bilstm_ctc.yml




