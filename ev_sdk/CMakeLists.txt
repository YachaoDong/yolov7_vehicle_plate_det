cmake_minimum_required(VERSION 3.5)
project(ji)

set(CMAKE_CXX_STANDARD 11)
# 设置安装路径前缀
set(CMAKE_INSTALL_PREFIX "/usr/local/ev_sdk")
# 设置lib安装目录
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# 设置不含调试debug
SET(CMAKE_BUILD_TYPE "Release")

# 设置gdb调试
# SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS}  -O0 -Wall -g2 -ggdb")

#先编译三方库中的源码，并包含头文件路径
#jsoncpp相关头文件路径和源文件
include_directories(${CMAKE_SOURCE_DIR}/3rd/jsoncpp_simple/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd/jsoncpp_simple)

#配置点线框解析相关头文件路径和源文件
include_directories(${CMAKE_SOURCE_DIR}/3rd/wkt_parser/include)
#CMAKE_SOURCE_DIR: /project/ev_sdk
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd/wkt_parser)


# yolo_det
include_directories(
    ${CMAKE_SOURCE_DIR}/src/yolo_src
    ${CMAKE_SOURCE_DIR}/src/yolo_src/include
    ${CMAKE_SOURCE_DIR}/src/yolo_src/application
    ${CMAKE_SOURCE_DIR}/src/yolo_src/tensorRT
    ${CMAKE_SOURCE_DIR}/src/yolo_src/tensorRT/common
)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/yolo_src)


# paddleocr link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
# TODO 没有连接完全的include，应该按照paddle_src camkelist中的进行include
include_directories(
    ${CMAKE_SOURCE_DIR}/src/paddle_src/include 
    ${CMAKE_SOURCE_DIR}/3rd/paddle_inference/paddle/include
)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/paddle_src)




# 使用pkg-config查找必要的依赖包
find_package(OpenCV REQUIRED)
find_library(glog REQUIRED)
# ENV{} 读取环境变量
set(ENV{PKG_CONFIG_PATH} "ENV{PKG_CONFIG_PATH}")
find_package(PkgConfig REQUIRED)



# 包含头文件
include_directories(/usr/local/cuda-11.1/targets/x86_64-linux/include/)

# 设置安装路径
# CMAKE_CURRENT_SOURCE_DIR: /project/ev_sdk
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:/usr/local/ev_sdk/lib")

# 编译libji.so
#SRCS: /project/ev_sdk/src/SampleAlgorithm.cpp;/project/ev_sdk/src/ji.cpp
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRCS)
# 将${SRCS}添加到 libji_SOURCES中
list(APPEND libji_SOURCES  ${SRCS})

add_library (${PROJECT_NAME} SHARED ${libji_SOURCES})

target_include_directories(${PROJECT_NAME}
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        #PUBLIC ${CMAKE_SOURCE_DIR}/src/paddle_src
        #PUBLIC ${CMAKE_SOURCE_DIR}/3rd/paddle_inference/third_party/install/xxhash/include
        )
# -L 指定库文件目录 -l 库文件名
# nvinfer nvonnxparser -L/usr/local/cuda/lib64/ -lcudart          
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        glog
        WKTParser
        jsoncpp
        yolodet 
        plateocr
        -L/usr/local/cuda/lib64/ -lcudart
        -L/usr/local/ev_sdk/3rd/paddle_inference/third_party/install/xxhash/lib/ -lxxhash
        -L/usr/local/ev_sdk/lib/ -lprotobuf
        )


# Install libji.so
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/3rd/fonts DESTINATION lib)
# ---  END EV_SDK DEPENDENCIES  ---
#         -L/usr/local/ev_sdk/lib/ -lyolodet
#        -L/usr/local/ev_sdk/lib/ -lplateocr
#        -L/usr/local/cuda/lib64/ -lcudart