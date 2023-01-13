/*
 * Copyright (c) 2021 ExtremeVision Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * 本代码实现一个车辆车牌检测报警的的实例,判断对应配置的ROI内是否存在车辆,并产生对应的图像和json返回值
 * 算法实例中展示了如何利用ev_sdk中提供的一些库和工具源码快速完成算法的开发
 * 本算法实例中有一个管理算法配置的配置对象mConfig,用于解析配置文件,并根据传入的配置字符串动态更新ROI
 * 本算法实例中有一个基于YOLOX的目标检测器实例,并采用Tensorrt用来实现模型的推理
 * 本算法利用ji_utils.h中的工具函数实现绘图功能
 * 本算法利用三方库中的wkt工具函数实现roi的解析
 * 本算法利用三方库中的jsoncpp工具实现json的解析和格式化输出
 * 新添加的模型推理,跟踪等功能最好以对象成员的方式添加到算法类中,不要将过多的更能添加到同一个类中
 **/

#ifndef JI_SAMPLEALGORITHM_HPP
#define JI_SAMPLEALGORITHM_HPP
#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ji.h"
#include "Configuration.hpp"
// #include "SampleDetector.h"

// TODO include 路径
// yolodet
//#include "yolo_src/yolodet.h"
#include "yolodet.h"

// paddleocr
//#include "paddle_src/include/paddleocr.h"
//#include "paddleocr.h"
#include "ocr_rec.h"
#include "preprocess_op.h"
#include "utility.h"

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <paddle_api.h>
#include <paddle_inference_api.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>



#define STATUS int
using namespace std;
using namespace cv;
using namespace Yolov7Det;

using namespace PaddleOCR;
using namespace paddle_infer;


typedef struct PlateInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float x4;
    float y4;
    
    float confidence;
    int class_label;
    int color_label;
    string ocr_text;
} PlateInfo;

class SampleAlgorithm
{

public:
    
    SampleAlgorithm();
    ~SampleAlgorithm();

    /*
     * @breif 初始化算法运行资源
     * @return STATUS 返回调用结果,成功返回STATUS_SUCCESS
    */    
    STATUS Init();

    /*
     * @breif 去初始化，释放算法运行资源
     * @return STATUS 返回调用结果,成功返回STATUS_SUCCESS
    */    
    STATUS UnInit();

    /*
     * @breif 算法业务处理函数，输入分析图片，返回算法分析结果
     * @param inFrame 输入图片对象  
     * @param args 输入算法参数, json字符串
     * @param event 返回的分析结果结构体
     * @return 返回结果, STATUS_SUCCESS代表调用成功
    */    
    STATUS Process(const Mat &inFrame, const char *args, JiEvent &event);

    /*
     * @breif 更新算法实例的配置
     * @param args 输入算法参数, json字符串     
     * @return 返回结果, STATUS_SUCCESS代表调用成功
    */ 
    STATUS UpdateConfig(const char *args);

    
    /*
     * @breif 调用Process接口后,获取处理后的图像
     * @param out 返回处理后的图像结构体     
     * @param outCount 返回调用次数的计数
     * @return 返回结果, STATUS_SUCCESS代表调用成功
    */ 
    STATUS GetOutFrame(JiImageInfo **out, unsigned int &outCount);
    
    
    // ocr
//     std::vector<std::vector<OCRPredictResult>>
//         mOCR(cv::Mat srcimg, bool det, bool rec,
//                    bool cls) {
//         std::vector<double> time_info_det = {0, 0, 0};
//         std::vector<double> time_info_rec = {0, 0, 0};
//         std::vector<double> time_info_cls = {0, 0, 0};
//         std::vector<std::vector<OCRPredictResult>> ocr_results;
//         if (!det) {
//             std::vector<OCRPredictResult> ocr_result;
//             // read image
//             std::vector<cv::Mat> img_list;
//             img_list.push_back(srcimg);
//             OCRPredictResult res;
//             ocr_result.push_back(res);
//             if (rec) {
//                 this->mREC(img_list, ocr_result, time_info_rec);
//             }
//             for (int i = 0; i < 1; ++i) {
//                 std::vector<OCRPredictResult> ocr_result_tmp;
//                 ocr_result_tmp.push_back(ocr_result[i]);
//                 ocr_results.push_back(ocr_result_tmp);
//             }
//         }
//         return ocr_results;
//     } 
    
    

    

private:
    cv::Mat mOutputFrame{0};    // 用于存储算法处理后的输出图像，根据ji.h的接口规范，接口实现需要负责释放该资源    
    JiImageInfo mOutImage[1];
    unsigned int mOutCount = 1;//本demo每次仅分析处理一幅图    
    Configuration mConfig;     //跟配置相关的类

public:
    //接口的返回值的定义
    static const int ERROR_BASE = 0x0100;
    static const int ERROR_INPUT = 0x0101;
    static const int ERROR_INIT = 0x0102;
    static const int ERROR_PROCESS = 0x0103;
    static const int ERROR_CONFIG = 0x0104;
    static const int STATUS_SUCCESS = 0x0000;

    
       
private:
    std::string mStrLastArg;  //算法参数缓存,动态参数与缓存参数不一致时才会进行更新  
    std::string mStrOutJson;  //返回的json缓存,注意当算法实例销毁时,对应的通过算法接口获取的json字符串也将不在可用
    //std::shared_ptr<SampleDetector> mDetector{nullptr}; //算法检测器实例
    
    // 报警信息特征
    std::string alert_text = "";
    // 存放alert_text的vector，用于判断是否报警
    std::vector<std::string> pred_frames_vector;
    
    // 获得连续相同元素的长度
    int getMaxSameSeq(std::vector<std::string> pred_frames_vector, int &max_i);

    // 对vector<Yolo::BoxArray>按照conf从高到低排序
    static bool GreaterSort (Yolo::Box a, Yolo::Box b) {
        return (a.confidence > b.confidence);
    }

    // 分割字符串
    void stringSplit(const string& s, vector<string>& tokens, char delim='_');


    // ocr 推理时识别调用
    std::vector<std::vector<OCRPredictResult>> mOCR(cv::Mat srcimg, bool det, bool rec, bool cls);

    // rec 被mOCR调用，推理代码中不调用
    void mREC(std::vector<cv::Mat> img_list,
              std::vector<OCRPredictResult> &ocr_results,
              std::vector<double> &times);
    
    
    std::shared_ptr<YOLODET> mDetector{nullptr}; //yolo det实例
    CRNNRecognizer *mRecognizer = nullptr;   // rec 初始化， init时加载模型
    
};

#endif //JI_SAMPLEALGORITHM_HPP
