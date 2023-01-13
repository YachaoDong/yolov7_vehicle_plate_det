cmake_minimum_required(VERSION 2.6)
project(yolodet)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE Release)

# 设置gdb调试
#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")



# 如果你是不同显卡，请设置为显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
#set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")

# 如果你的opencv找不到，可以自己指定目录
#set(OpenCV_DIR   "/project/train/src_repo/paddle_cpp_infer/opencv-3.4.7/opencv3/")


set(CUDA_TOOLKIT_ROOT_DIR     "/usr/local/cuda")

# change 包含tebsorrt lib cudnnlib
set(CUDNN_INCLUDE_DIR    "/usr/include")
set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
# 

# 因为protobuf，需要用特定版本，所以这里指定路径
set(PROTOBUF_DIR "/usr/local/ev_sdk/3rd/protobuf-3.11.4-install")



find_package(CUDA REQUIRED)
find_package(OpenCV)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/application
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorRT
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorRT/common
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${PROTOBUF_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
    ${CUDNN_INCLUDE_DIR}
)

# 切记，protobuf的lib目录一定要比tensorRT目录前面，因为tensorRTlib下带有protobuf的so文件
# 这可能带来错误
link_directories(
    ${PROTOBUF_DIR}/lib
    ${TENSORRT_LIB_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    #${CUDNN_DIR}/lib64
)



set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
file(GLOB_RECURSE cpp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE cuda_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
cuda_add_library(plugin_list SHARED ${cuda_srcs})

target_link_libraries(plugin_list nvinfer nvinfer_plugin)
target_link_libraries(plugin_list cuda cublas cudart cudnn)
target_link_libraries(plugin_list protobuf pthread)
target_link_libraries(plugin_list ${OpenCV_LIBS})

# add_executable(pro ${cpp_srcs}) #del
add_library (yolodet SHARED ${cpp_srcs})

# 如果提示插件找不到，请使用dlopen(xxx.so, NOW)的方式手动加载可以解决插件找不到问题
target_link_libraries(yolodet nvinfer nvinfer_plugin)
target_link_libraries(yolodet cuda cublas cudart cudnn)
target_link_libraries(yolodet protobuf pthread plugin_list)
target_link_libraries(yolodet ${OpenCV_LIBS})

# change
#target_compile_options(yolodet  PRIVATE -std=c++11 -Wall -O3 -fPIC)
# Install libyolodet.so

set(CMAKE_INSTALL_PREFIX "/usr/local/ev_sdk")
install(TARGETS yolodet DESTINATION lib)



