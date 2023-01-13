#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <glog/logging.h>

#include "reader.h"
#include "writer.h"
#include "value.h"
#include "ji_utils.h"
#include "SampleAlgorithm.hpp"

#define JSON_ALERT_FLAG_KEY ("is_alert")
#define JSON_ALERT_FLAG_TRUE true
#define JSON_ALERT_FLAG_FALSE false

// TODO 正确的include路径
// paddle ocr
// #include "opencv2/core.hpp"
// #include "opencv2/imgcodecs.hpp"
// #include "opencv2/imgproc.hpp"
// #include <include/args.h>
// #include <include/paddleocr.h>
#include <vector>

// yolo

using namespace std;
using namespace PaddleOCR;
using namespace Yolov7Det;


SampleAlgorithm::SampleAlgorithm()
{
}

SampleAlgorithm::~SampleAlgorithm()
{
    UnInit();
}

STATUS SampleAlgorithm::Init()
{
    // 从默认的配置文件读取相关配置参数
    const char *configFile = "/usr/local/ev_sdk/config/algo_config.json";
    SDKLOG(INFO) << "Parsing configuration file: " << configFile;
    std::ifstream confIfs(configFile);
    if (confIfs.is_open())
    {
        size_t len = getFileLen(confIfs);
        char *confStr = new char[len + 1];
        confIfs.read(confStr, len);
        confStr[len] = '\0';
    	SDKLOG(INFO) << "Configs:"<<confStr;
        mConfig.ParseAndUpdateArgs(confStr);
        delete[] confStr;
        confIfs.close();
    }

    // yolo model init
    mDetector = std::make_shared<YOLODET>();
    mDetector->DetInit(mConfig.algoConfig.gpu_id,
                      mConfig.algoConfig.det_engine_file,
                      mConfig.algoConfig.conf_thresh,
                      mConfig.algoConfig.nms_thresh);

    // ocr model init
    this->mRecognizer = new CRNNRecognizer(mConfig.algoConfig.rec_model_dir,
                  mConfig.algoConfig.rec_use_gpu,
                  mConfig.algoConfig.gpu_id,
                  mConfig.algoConfig.rec_gpu_mem,
                  mConfig.algoConfig.rec_cpu_math_library_num_threads,
                  mConfig.algoConfig.rec_use_mkldnn,
                  mConfig.algoConfig.rec_char_dict_path,
                  false,
                  mConfig.algoConfig.rec_precision,
                  mConfig.algoConfig.rec_batch_num,
                  mConfig.algoConfig.rec_img_h,
                  mConfig.algoConfig.rec_img_w);

    return STATUS_SUCCESS;
}


STATUS SampleAlgorithm::UnInit()
{
    // yolo detector uninit
    if(mDetector.get() != nullptr)
    {
        mDetector->DetUnInit();
        mDetector.reset();
    }


    // ocr uninit
    if (this->mRecognizer != nullptr) {
        delete this->mRecognizer;
        this->mRecognizer = nullptr;
    }

    return STATUS_SUCCESS;
}

STATUS SampleAlgorithm::UpdateConfig(const char *args)
{
    if (args == nullptr)
    {
        SDKLOG(ERROR) << "mConfig string is null ";
        return ERROR_CONFIG;
    }
    mConfig.ParseAndUpdateArgs(args);
    return STATUS_SUCCESS;
}

STATUS SampleAlgorithm::GetOutFrame(JiImageInfo **out, unsigned int &outCount)
{
    outCount = mOutCount;

    mOutImage[0].nWidth = mOutputFrame.cols;
    mOutImage[0].nHeight = mOutputFrame.rows;
    mOutImage[0].nFormat = JI_IMAGE_TYPE_BGR;
    mOutImage[0].nDataType = JI_UNSIGNED_CHAR;
    mOutImage[0].nWidthStride = mOutputFrame.step;
    mOutImage[0].pData = mOutputFrame.data;

    *out = mOutImage;
    return STATUS_SUCCESS;
}

STATUS SampleAlgorithm::Process(const cv::Mat &inFrame, const char *args, JiEvent &event)
{
    //输入图片为空的时候直接返回错误
    if (inFrame.empty())
    {
        SDKLOG(ERROR) << "Invalid input!";
        return ERROR_INPUT;
    }

    //由于roi配置是归一化的坐标,所以输出图片的大小改变时,需要更新ROI的配置
    if (inFrame.cols != mConfig.currentInFrameSize.width || inFrame.rows != mConfig.currentInFrameSize.height)
    {
	    SDKLOG(INFO)<<"Update ROI Info...";
        mConfig.UpdateROIInfo(inFrame.cols, inFrame.rows);
    }

    //如果输入的参数不为空且与上一次的参数不完全一致,需要调用更新配置的接口
    if(args != nullptr && mStrLastArg != args)
    {
    	mStrLastArg = args;
        SDKLOG(INFO) << "Update args:" << args;
        mConfig.ParseAndUpdateArgs(args);
    }

    // 针对整张图进行推理,获取所有的检测目标,并过滤出在ROI内的目标
    //std::vector<BoxInfo> detectedObjects;
    //std::vector<BoxInfo> validTargets;
    Yolo::BoxArray validTargets;
    Yolo::BoxArray validDetecedTargets;
    //std::vector<Yolo::BoxArray> validTargets;
    //std::vector<Yolo::BoxArray> validDetecedTargets;
    // 算法处理
    cv::Mat img = inFrame.clone();

    // yolo det model process
    //Yolo::BoxArray detectedObjects;
    auto detectedObjects = mDetector->commit_gpu(img);  // 得到的是vector<Box> [(left, top, right, bottom, confidence, class_label, color_label), ...]

    // origin model
    // mDetector->ProcessImage(img, detectedObjects, mConfig.algoConfig.thresh);

//     Yolo::Box my_plate = {971, 886, 1060, 908, 0.9, 11, 2};
//     detectedObjects.emplace_back(my_plate);


    // 结果按照conf 由高到低排序
    std::sort(detectedObjects.begin(), detectedObjects.end(), GreaterSort);
    for (auto &obj : detectedObjects)
    {
        for (auto &roiPolygon : mConfig.currentROIOrigPolygons)
        {
            int mid_x = (obj.left + obj.right) / 2;
            int mid_y = (obj.top + obj.bottom) / 2;
            // 当检测的目标的中心点在ROI内的话，就视为闯入ROI的有效目标
            if (WKTParser::inPolygon(roiPolygon, cv::Point(mid_x, mid_y)))
            {
                validTargets.emplace_back(obj);
            }
        }
    }

    SDKLOG_FIRST_N(INFO, 5) << "detected targets : " << detectedObjects.size() << " valid targets :  " << validTargets.size();

    // ocr 检测，
    PlateInfo plate_info;
    std::vector<PlateInfo> validOcrObjects;
    for (auto &obj : validTargets){
        // 当为有效目标时 且 检测目标为车牌时， 使用ocr检测
        if ((obj.class_label==11) || (obj.class_label==12)){
//             continue;
            // plate "NONE" 判断
            int plate_w = (int)(obj.right - obj.left);
            int plate_h = (int)(obj.bottom - obj.top);
            int plate_wh = plate_w * plate_h;
            if((plate_w < 33) || (plate_h < 17) || (plate_wh < 576)){
                plate_info.ocr_text = "NONE";
            }
            //车牌识别
            else{
                // 车牌图像裁剪
                cv::Mat ori_ocrimg;
                inFrame.copyTo(ori_ocrimg);

                cv::Mat ocr_img;

                int ori_img_w = ori_ocrimg.size().width;
                int ori_img_h = ori_ocrimg.size().height;

                obj.left = std::max(0, int(obj.left));
                obj.top = std::max(0, int(obj.top));

                obj.right = std::min(ori_img_w, int(obj.right));
                obj.bottom = std::min(ori_img_h, int(obj.bottom));

                ori_ocrimg(cv::Rect(obj.left, obj.top, obj.right - obj.left, obj.bottom - obj.top)).copyTo(ocr_img);
                // ocr推理
                std::vector<std::vector<OCRPredictResult>> ocr_results = this->mOCR(ocr_img, false, true, false);
                plate_info.ocr_text = ocr_results[0][0].text; // rec_score= .score

            }
            // x_min, y_min, x_max, y_max = map(int, [box.left, box.top, box.right, box.bottom])
            plate_info.x1 = obj.left;
            plate_info.y1 = obj.top;
            plate_info.x2 = obj.right;
            plate_info.y2 = obj.top;
            plate_info.x3 = obj.right;
            plate_info.y3 = obj.bottom;
            plate_info.x4 = obj.left;
            plate_info.y4 = obj.bottom;
            plate_info.confidence = obj.confidence;
            plate_info.class_label = obj.class_label;
            plate_info.color_label = obj.color_label;

            validOcrObjects.emplace_back(plate_info);
        }
        // 筛选出 在roi区域中vehilce的目标
        else{
            validDetecedTargets.emplace_back(obj);
        }
    } // ocr检测

    // 创建输出图
    inFrame.copyTo(mOutputFrame);
    // 画ROI区域
    if (mConfig.drawROIArea && !mConfig.currentROIOrigPolygons.empty())
    {
        drawPolygon(mOutputFrame, mConfig.currentROIOrigPolygons, cv::Scalar(mConfig.roiColor[0], mConfig.roiColor[1], mConfig.roiColor[2]),
                    mConfig.roiColor[3], cv::LINE_AA, mConfig.roiLineThickness, mConfig.roiFill);
    }
    //并将检测到的在ROI内部的目标画到图上 (left, top, right, bottom, confidence, class_label, color_label),
    // 画除车牌以外的图      目标汇总：validDetecedTargets + validOcrObjects = validTargets +ROI = detectedObjects
    for (auto &object : validDetecedTargets)
    {
        if (mConfig.drawResult)
        {
            std::stringstream ss;
            // color
            ss << (object.color_label > mConfig.targetColorTextMap[mConfig.language].size() - 1 ? "":
                   mConfig.targetColorTextMap[mConfig.language][object.color_label])<< " ";
            // obj
            ss << (object.class_label > mConfig.targetRectTextMap[mConfig.language].size() - 1 ? "":
                   mConfig.targetRectTextMap[mConfig.language][object.class_label]);
            if (mConfig.drawConfidence)
            {
                ss.precision(0);
                ss <<std::fixed << (object.class_label > mConfig.targetRectTextMap[mConfig.language].size() - 1 ? "" : ": ")
                    //<< (object.color_label > mConfig.targetColorTextMap[mConfig.language].size() - 1 ? "" : ": ")
                    << object.confidence * 100<< "%";
            }
            cv::Rect rect = cv::Rect{object.left, object.top, object.right - object.left, object.bottom - object.top};
            drawRectAndText(mOutputFrame, rect, ss.str(), mConfig.targetRectLineThickness, cv::LINE_AA,
                            cv::Scalar(mConfig.targetRectColor[0], mConfig.targetRectColor[1], mConfig.targetRectColor[2]), mConfig.targetRectColor[3], mConfig.targetTextHeight,
                            cv::Scalar(mConfig.textFgColor[0], mConfig.textFgColor[1], mConfig.textFgColor[2]),
                            cv::Scalar(mConfig.textBgColor[0], mConfig.textBgColor[1], mConfig.textBgColor[2]));
        }
    }
    // 画车牌图 ocr
    for (auto &object : validOcrObjects)
    {
        if (mConfig.drawResult)
        {
            std::stringstream ss;
            // color
            ss << (object.color_label > mConfig.targetColorTextMap[mConfig.language].size() - 1 ? "":
                   mConfig.targetColorTextMap[mConfig.language][object.color_label])<<" ";
            // obj
            ss << (object.class_label > mConfig.targetRectTextMap[mConfig.language].size() - 1 ? "":
                   mConfig.targetRectTextMap[mConfig.language][object.class_label])<<" ";
            // ocr text
            ss << object.ocr_text << " ";

            if (mConfig.drawConfidence)
            {
                ss.precision(0);
                ss << std::fixed <<(object.class_label > mConfig.targetRectTextMap[mConfig.language].size() - 1 ? "" : ": ")
                    //<<(object.color_label > mConfig.targetColorTextMap[mConfig.language].size() - 1 ? "" : ": ")
                    << object.confidence * 100 << "%";
            }
            cv::Rect rect = cv::Rect{object.x1, object.y1, object.x3 - object.x1, object.y3 - object.y1};
            drawRectAndText(mOutputFrame, rect, ss.str(), mConfig.targetRectLineThickness, cv::LINE_AA,
                            cv::Scalar(mConfig.targetRectColor[0], mConfig.targetRectColor[1], mConfig.targetRectColor[2]), mConfig.targetRectColor[3], mConfig.targetTextHeight,
                            cv::Scalar(mConfig.textFgColor[0], mConfig.textFgColor[1], mConfig.textFgColor[2]),
                            cv::Scalar(mConfig.textBgColor[0], mConfig.textBgColor[1], mConfig.textBgColor[2]));
        }
    } //画图


    // 业务报警逻辑：  TODO: 这里是取的第一个目标validDetecedTargets[0]作为判断
    alert_text  = mConfig.Cid;
    for (auto &object : validDetecedTargets){
        //alert_text = alert_text  + mConfig.Cid;
        // 车辆类型控制报警
        if (mConfig.vehicleClassAlert){
            alert_text = alert_text + "_" + (object.class_label > mConfig.targetRectTextMap["en"].size() - 1 ?
                                                     "obj": mConfig.targetRectTextMap["en"][object.class_label]);
        }
        // 车辆颜色控制报警
        if (mConfig.vehicleColorAlert){
            alert_text = alert_text + "_" + (object.color_label > mConfig.targetColorTextMap["en"].size() - 1 ?
                                                         "obj": mConfig.targetColorTextMap["en"][object.color_label]);
        }
        break; //只取第一个值
    }

    for (auto &object : validOcrObjects){
        //alert_text = alert_text  + "_" + mConfig.Cid;
        // 车牌内容控制报警
        if (mConfig.plateOcrAlert){
            alert_text = alert_text + "_" + object.ocr_text;
        }
        // 车牌颜色控制报警
        if (mConfig.plateColorAlert){
            alert_text = alert_text  + "_" + (object.color_label > mConfig.targetColorTextMap["en"].size() - 1 ?
                                                          "obj": mConfig.targetColorTextMap["en"][object.color_label]);
        }
        break; //只取第一个值
    }

    std::vector<std::string> alertInfoElem;
    Json::Value jAlgoValue;
    jAlgoValue["alert_vehicle_exclude"].resize(0);
    jAlgoValue["alert_vehicle_color_exclud"].resize(0);
    jAlgoValue["alert_plate_chinese_exclude"].resize(0);
    jAlgoValue["alert_plate_color_exclude"].resize(0);
    bool isNeedAlert = false; // 是否需要报警
    // 求最大连续子序列的长度，然后判断是否报警 ,然后pred_frames_vector = {}
    if (alert_text != mConfig.Cid){
        pred_frames_vector.emplace_back(alert_text);
    }
    int max_i = 0;
    int max_count = getMaxSameSeq(pred_frames_vector, max_i);
    std::cout<< "max_count:" << max_count << std::endl;
    if (max_count > mConfig.alertCountThreshold){
        isNeedAlert = true;
        // alertInfo：报警信息文本，即alert_text
        string alertInfo = pred_frames_vector[max_i];
        std::cout<< "alertInfo:" << alertInfo << std::endl;

        // 清空vector元素
        pred_frames_vector.clear();
        // 释放vector内存
        std::vector<std::string>().swap(pred_frames_vector);

        // 对alertInfo 字符串分割
        std::vector<string> alertInfoElem;
        stringSplit(alertInfo, alertInfoElem);

        //if (std::find(mConfig.alertVehicleExclude.begin(),  mConfig.alertVehicleExclude.end(), key) != v.end())

        // 判断alertInfo split _ 与exclude的交集
        for (auto &obj : mConfig.alertVehicleExclude){
            jAlgoValue["alert_vehicle_exclude"].append(obj);

            if (std::find(alertInfoElem.begin(),  alertInfoElem.end(), obj) != alertInfoElem.end()){
                isNeedAlert = false;
            }
        }
        for (auto &obj : mConfig.alertVehicleColorExclude){
            jAlgoValue["alert_vehicle_color_exclud"].append(obj);
            if (std::find(alertInfoElem.begin(),  alertInfoElem.end(), obj) != alertInfoElem.end()){
                isNeedAlert = false;
            }
        }
        // TODO 获取ocrtext的汉字
        for (auto &obj : mConfig.alertPlateChineseExclude){
            jAlgoValue["alert_plate_chinese_exclude"].append(obj);

            if(alertInfo.find(obj)!= string :: npos){
                isNeedAlert = false;
            }
        }
        for (auto &obj : mConfig.alertPlateColorExclude){
            jAlgoValue["alert_plate_color_exclude"].append(obj);
            if (std::find(alertInfoElem.begin(),  alertInfoElem.end(), obj) != alertInfoElem.end()){
                isNeedAlert = false;
            }
        }

    }
    //pred_frames_vector.clear();

    // 在图上画出报警
    if (isNeedAlert && mConfig.drawWarningText)
    {
        drawText(mOutputFrame, mConfig.warningTextMap[mConfig.language], mConfig.warningTextSize,
                 cv::Scalar(mConfig.warningTextFg[0], mConfig.warningTextFg[1], mConfig.warningTextFg[2]),
                 cv::Scalar(mConfig.warningTextBg[0], mConfig.warningTextBg[1], mConfig.warningTextBg[2]), mConfig.warningTextLeftTop);
    }

    // TODO 报警信息target_info


    // TODO 多个ROI如何判断处理



    // 将结果封装成json字符串
    bool jsonAlertCode = JSON_ALERT_FLAG_FALSE;
    if (isNeedAlert)
    {
        jsonAlertCode = JSON_ALERT_FLAG_TRUE;
    }
    Json::Value jRoot;
    //Json::Value jAlgoValue;   // algorithm_data
    Json::Value jDetectValue; // model_data

// ---create algorithm_data----------------
    // "is_alert"
    jAlgoValue[JSON_ALERT_FLAG_KEY] = jsonAlertCode;
    // "alert_count_threshold"
    jAlgoValue["alert_count_threshold"] = mConfig.alertCountThreshold;
    jAlgoValue["cid"] = mConfig.Cid;
    // TODO
    jAlgoValue["target_count"] = 0;
    jAlgoValue["target_vehicle_count"] = 0;
    jAlgoValue["target_plate_count"] = 0;

    jAlgoValue["vehicle_class_alert"] = mConfig.vehicleClassAlert;
    jAlgoValue["plate_ocr_alert"] = mConfig.plateOcrAlert;
    jAlgoValue["vehicle_color_alert"] = mConfig.vehicleColorAlert;
    jAlgoValue["plate_color_alert"] = mConfig.plateOcrAlert;

    // 报警信息控制字段写入json
//     for (auto &obj : mConfig.alertVehicleExclude){
//         jAlgoValue["alert_vehicle_exclude"].append(obj);
//     }
//     for (auto &obj : mConfig.alertVehicleColorExclude){
//         jAlgoValue["alert_vehicle_color_exclud"].append(obj);
//     }
//     for (auto &obj : mConfig.alertPlateChineseExclude){
//         jAlgoValue["alert_plate_chinese_exclude"].append(obj);
//     }
//     for (auto &obj : mConfig.alertPlateColorExclude){
//         jAlgoValue["alert_plate_color_exclude"].append(obj);
//     }

    // TODO 报警信息
    jAlgoValue["target_info"].resize(0);
    // vehocel obj
    if (isNeedAlert){
        // vehicle obj
        if (mConfig.vehicleClassAlert){
            if (!validDetecedTargets.empty()){
                jAlgoValue["target_vehicle_count"] = 1;
                auto obj = validDetecedTargets[0];
                Json::Value jTmpValue;
                jTmpValue["x"] = int(obj.left);
                jTmpValue["y"] = int(obj.top);
                jTmpValue["width"] = int(obj.right - obj.left);
                jTmpValue["height"] = int(obj.bottom - obj.top);
                jTmpValue["name"] = (obj.class_label > mConfig.targetRectTextMap["en"].size() - 1 ?
                                     "obj": mConfig.targetRectTextMap["en"][obj.class_label]);

                jTmpValue["color"] = (obj.color_label > mConfig.targetColorTextMap["en"].size() - 1 ?
                                                         "obj": mConfig.targetColorTextMap["en"][obj.color_label]);

                jTmpValue["confidence"] = obj.confidence;

                jAlgoValue["target_info"].append(jTmpValue);
            }
        }

        // ocr obj
        if (mConfig.plateOcrAlert){
            if (!validOcrObjects.empty()){
                jAlgoValue["target_plate_count"] = 1;
                auto obj = validOcrObjects[0];
                Json::Value jTmpValue;
                jTmpValue["points"].append(int(obj.x1));
                jTmpValue["points"].append(int(obj.y1));
                jTmpValue["points"].append(int(obj.x2));
                jTmpValue["points"].append(int(obj.y2));
                jTmpValue["points"].append(int(obj.x3));
                jTmpValue["points"].append(int(obj.y3));
                jTmpValue["points"].append(int(obj.x4));
                jTmpValue["points"].append(int(obj.y4));

                jTmpValue["confidence"] = obj.confidence;
                jTmpValue["ocr"] = obj.ocr_text;
                jTmpValue["name"] = (obj.class_label > mConfig.targetRectTextMap["en"].size() - 1 ?
                                     "obj": mConfig.targetRectTextMap["en"][obj.class_label]);

                jTmpValue["color"] = (obj.color_label > mConfig.targetColorTextMap["en"].size() - 1 ?
                                      "obj": mConfig.targetColorTextMap["en"][obj.color_label]);

                jAlgoValue["target_info"].append(jTmpValue);
            }
        }
    }

    int target_vehicle_count = jAlgoValue.get("target_vehicle_count", 0).asInt();
    int target_plate_count = jAlgoValue.get("target_plate_count", 0).asInt();
    jAlgoValue["target_count"] = target_vehicle_count + target_plate_count;

    jRoot["algorithm_data"] = jAlgoValue;
// ---create algorithm_data---


// ---create model data---
    jDetectValue["objects"].resize(0);
    // vehocel obj
    for (auto &obj : validDetecedTargets)
    {
        // vehocel obj
        if((obj.class_label != 11) && (obj.class_label != 12)){
            Json::Value jTmpValue;
            jTmpValue["x"] = int(obj.left);
            jTmpValue["y"] = int(obj.top);
            jTmpValue["width"] = int(obj.right - obj.left);
            jTmpValue["height"] = int(obj.bottom - obj.top);
            jTmpValue["name"] = (obj.class_label > mConfig.targetRectTextMap["en"].size() - 1 ? "obj": mConfig.targetRectTextMap["en"][obj.class_label]);

            jTmpValue["color"] = (obj.color_label > mConfig.targetColorTextMap["en"].size() - 1 ?
                                  "obj": mConfig.targetColorTextMap["en"][obj.color_label]);

            jTmpValue["confidence"] = obj.confidence;

            jDetectValue["objects"].append(jTmpValue);
        }
    }
    // plate obj
    for (auto &obj : validOcrObjects)
    {
        Json::Value jTmpValue;
        jTmpValue["points"].append(int(obj.x1));
        jTmpValue["points"].append(int(obj.y1));
        jTmpValue["points"].append(int(obj.x2));
        jTmpValue["points"].append(int(obj.y2));
        jTmpValue["points"].append(int(obj.x3));
        jTmpValue["points"].append(int(obj.y3));
        jTmpValue["points"].append(int(obj.x4));
        jTmpValue["points"].append(int(obj.y4));

        jTmpValue["confidence"] = obj.confidence;
        jTmpValue["ocr"] = obj.ocr_text;
        jTmpValue["name"] = (obj.class_label > mConfig.targetRectTextMap[mConfig.language].size() - 1 ? "obj":
                             mConfig.targetRectTextMap[mConfig.language][obj.class_label]);

        jTmpValue["color"] = (obj.color_label > mConfig.targetColorTextMap["en"].size() - 1 ?
                              "obj": mConfig.targetColorTextMap["en"][obj.color_label]);

        jDetectValue["objects"].append(jTmpValue);
    }
    jRoot["model_data"] = jDetectValue;

// ---create model data---


    Json::StreamWriterBuilder writerBuilder;
    writerBuilder.settings_["precision"] = 2;
    writerBuilder.settings_["emitUTF8"] = true;
    std::unique_ptr<Json::StreamWriter> jsonWriter(writerBuilder.newStreamWriter());
    std::ostringstream os;
    jsonWriter->write(jRoot, &os);
    mStrOutJson = os.str();
    // 注意：JiEvent.code需要根据需要填充，切勿弄反
    if (isNeedAlert)
    {
        event.code = JISDK_CODE_ALARM;
    }
    else
    {
        event.code = JISDK_CODE_NORMAL;
    }
    event.json = mStrOutJson.c_str();
    return STATUS_SUCCESS;
}





// 获得连续相同元素的长度
int SampleAlgorithm::getMaxSameSeq(std::vector<std::string> pred_frames_vector, int &max_i)
{
  int count1=0,count2=0;
  for(int i=0; i<pred_frames_vector.size(); i++)
  {
    std::cout << "pred_frames_vector:" << pred_frames_vector[i] <<endl;
    int j=i;
    while(j < pred_frames_vector.size())
    {
        if(pred_frames_vector[i]==pred_frames_vector[j])
            count1++;
        else{
            break;
        }
        j++;
    }
    if(count2<count1)
    {
        count2=count1;
        max_i = j-1;
    }
    count1=0;
  }
  return count2;
}

// 分割字符串
void SampleAlgorithm::stringSplit(const string& s, vector<string>& tokens, char delim){
    tokens.clear();
    auto string_find_first_not = [s, delim](size_t pos = 0) -> size_t {
        for (size_t i = pos; i < s.size(); i++) {
            if (s[i] != delim) return i;
        }
        return string::npos;
    };
    size_t lastPos = string_find_first_not(0);
    size_t pos = s.find(delim, lastPos);
    while (lastPos != string::npos) {
        std::cout<<"s.substr(lastPos, pos - lastPos):" <<s.substr(lastPos, pos - lastPos) <<std::endl;
        tokens.emplace_back(s.substr(lastPos, pos - lastPos));
        lastPos = string_find_first_not(pos);
        pos = s.find(delim, lastPos);
    }
}

// ocr 推理时识别调用
std::vector<std::vector<OCRPredictResult>>
    SampleAlgorithm::mOCR(cv::Mat srcimg, bool det, bool rec,
               bool cls) {
    std::vector<double> time_info_det = {0, 0, 0};
    std::vector<double> time_info_rec = {0, 0, 0};
    std::vector<double> time_info_cls = {0, 0, 0};
    std::vector<std::vector<OCRPredictResult>> ocr_results;
    if (!det) {
        std::vector<OCRPredictResult> ocr_result;
        // read image
        std::vector<cv::Mat> img_list;
        img_list.push_back(srcimg);
        OCRPredictResult res;
        ocr_result.push_back(res);
        if (rec) {
            this->mREC(img_list, ocr_result, time_info_rec);
        }
        for (int i = 0; i < 1; ++i) {
            std::vector<OCRPredictResult> ocr_result_tmp;
            ocr_result_tmp.push_back(ocr_result[i]);
            ocr_results.push_back(ocr_result_tmp);
        }
    }
    return ocr_results;
}


// rec 被mOCR调用，推理代码中不调用
void SampleAlgorithm::mREC(std::vector<cv::Mat> img_list,
            std::vector<OCRPredictResult> &ocr_results,
            std::vector<double> &times) {
  std::vector<std::string> rec_texts(img_list.size(), "");
  std::vector<float> rec_text_scores(img_list.size(), 0);
  std::vector<double> rec_times;
  this->mRecognizer->Run(img_list, rec_texts, rec_text_scores, rec_times);
  // output rec results
  for (int i = 0; i < rec_texts.size(); i++) {
    ocr_results[i].text = rec_texts[i];
    ocr_results[i].score = rec_text_scores[i];
  }
}





