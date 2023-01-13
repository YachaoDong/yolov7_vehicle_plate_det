# bash /usr/local/ev_sdk/gen_trt.sh

# local: /usr/local/ev_sdk/model1/35403
# test env:/project/train/models/exp/weights/deploy_yolov7_best.onnx
#  /project/train/models/paddle_infer

cd /project/train/models/exp/weights
# rm last.pt best.pt

# best.pt 2 best.trt
# cp /usr/local/ev_sdk/config/algo_config_1.json /usr/local/ev_sdk/config/algo_config.json
# /usr/local/ev_sdk/bin/test-ji-api -f 1 -i /project/train/src_repo/1.jpg   -o /project/train/src_repo/out.jpg


# 1.先重参数化权重，并转为onnx
bash /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/rep_onnx.sh

# last.pt 2 last.trt
cp /usr/local/ev_sdk/config/algo_config_2.json /usr/local/ev_sdk/config/algo_config.json
/usr/local/ev_sdk/bin/test-ji-api -f 1 -i /project/train/src_repo/1.jpg  -o /project/train/src_repo/out.jpg
# 更改名字
cd /project/train/models/exp/weights
mv deploy_yolov7_last.trt  last_200.trt
rm *.pt
rm *.onnx
rm -r /project/train/models/paddle_infer/