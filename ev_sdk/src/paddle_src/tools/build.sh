OPENCV_DIR=/project/train/src_repo/paddle_cpp_infer/opencv-3.4.7/opencv3
LIB_DIR=/project/train/src_repo/paddle_inference
CUDA_LIB_DIR=/usr/local/cuda/lib64
CUDNN_LIB_DIR=/project/train/src_repo/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib
TENSORRT_DIR=/project/train/src_repo/TensorRT-8.4.1.5

BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DWITH_GPU=ON \
    -DWITH_STATIC_LIB=OFF \
    -DWITH_TENSORRT=ON \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \
    -DTENSORRT_DIR=${TENSORRT_DIR} \

make -j8
