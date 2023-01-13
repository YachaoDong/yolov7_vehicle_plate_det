# bash /usr/local/ev_sdk/for_new_instance.sh
# 用于重建实例，重新下载 cudnn tensorrt paddle_inference

# 1. cudnn
cd /project/train/src_repo
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-4467-files/0b9f2718-5c24-4a26-bc07-4b5f0e3305ef/cudnn-11.2-linux-x64-v8.1.1.33.tgz
tar -xvf cudnn-11.2-linux-x64-v8.1.1.33.tgz
cp cuda/include/*.h /usr/include/
cp cuda/lib64/libcudnn* /usr/lib/x86_64-linux-gnu/

# 2. tensorrt
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-4467-files/41bf036e-0f67-4cea-ae9e-00404d7e7e52/TensorRT-8.4.1.5.tar.gz
tar -xvf TensorRT-8.4.1.5.tar.gz
cp TensorRT-8.4.1.5/include/*.h /usr/include/x86_64-linux-gnu/
cp TensorRT-8.4.1.5/lib/lib* /usr/lib/x86_64-linux-gnu/
# 删除原文件
rm cudnn-11.2-linux-x64-v8.1.1.33.tgz TensorRT-8.4.1.5.tar.gz
rm -rf cuda/ TensorRT-8.4.1.5/


# 3. paddle_inference
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-4467-files/93f199c7-5099-4950-b70c-cfe68efd9070/paddle_inference.tgz
tar -xvf paddle_inference.tgz
rm paddle_inference.tgz
mv paddle_inference/ /usr/local/ev_sdk/3rd/

# 4. protobuf

wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-4467-files/da8dbaa3-2692-47ae-b173-34e65cc70151/protobuf-3.11.4-install.zip

unzip protobuf-3.11.4-install.zip
rm protobuf-3.11.4-install.zip
mv protobuf-3.11.4-install/ /usr/local/ev_sdk/3rd/






# 移动缺失库
# paddle_inference
# 节省空间可以使用 mv
mv /usr/local/ev_sdk/3rd/paddle_inference/paddle/lib/libpaddle_inference.so  /usr/local/ev_sdk/lib/
ln -s /usr/local/ev_sdk/lib/libpaddle_inference.so   /usr/local/ev_sdk/3rd/paddle_inference/paddle/lib/libpaddle_inference.so

mv /usr/local/ev_sdk/3rd/protobuf-3.11.4-install/lib/libprotobuf.so.22 /usr/local/ev_sdk/lib/
ln -s /usr/local/ev_sdk/lib/libprotobuf.so.22     /usr/local/ev_sdk/3rd/protobuf-3.11.4-install/lib/libprotobuf.so.22

mv /usr/local/ev_sdk/3rd/paddle_inference/third_party/install/mklml/lib/libiomp5.so /usr/local/ev_sdk/lib/
ln -s /usr/local/ev_sdk/lib/libiomp5.so    /usr/local/ev_sdk/3rd/paddle_inference/third_party/install/mklml/lib/libiomp5.so

mv /usr/local/ev_sdk/3rd/paddle_inference/third_party/install/mkldnn/lib/libdnnl.so.2  /usr/local/ev_sdk/lib/
ln -s /usr/local/ev_sdk/lib/libdnnl.so.2     /usr/local/ev_sdk/3rd/paddle_inference/third_party/install/mkldnn/lib/libdnnl.so.2


mv /usr/local/ev_sdk/3rd/paddle_inference/third_party/install/onnxruntime/lib/libonnxruntime.so.1.11.1 /usr/local/ev_sdk/lib/
ln -s /usr/local/ev_sdk/lib/libonnxruntime.so.1.11.1 /usr/local/ev_sdk/3rd/paddle_inference/third_party/install/onnxruntime/lib/libonnxruntime.so.1.11.1

mv /usr/local/ev_sdk/3rd/paddle_inference/third_party/install/paddle2onnx/lib/libpaddle2onnx.so.0.9.9 /usr/local/ev_sdk/lib/
ln -s /usr/local/ev_sdk/lib/libpaddle2onnx.so.0.9.9 /usr/local/ev_sdk/3rd/paddle_inference/third_party/install/paddle2onnx/lib/libpaddle2onnx.so.0.9.9

# 另外需要把中间生成的 libplugin_list.so 移动到正确的路径
#  mv /usr/local/ev_sdk/build/src/yolo_src/libplugin_list.so 移动到正确的路径 /usr/local/lib/


# 需要把protobuf-3.11.4-install/ lib 也copy到 /usr/local/lib 中




