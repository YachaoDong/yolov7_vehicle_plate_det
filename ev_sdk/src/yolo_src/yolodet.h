#pragma once
//yolo det
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <string>
#include <iostream>
#include "application/app_yolo/yolo.hpp"
#include "application/app_yolo/multi_gpu.hpp"
using namespace std;


namespace Yolov7Det {

class YOLODET {
public:
  // 模型初始化
  YOLODET();
  ~YOLODET();
  void DetInit(int deviceid, const string& engine_file, float confidence_threshold=0.25, float nms_threshold=0.45);
  void DetUnInit();

  // 推理函数
  ObjectDetector::BoxArray commit_gpu(cv::Mat srcimg);

//  std::vector<std::vector<OCRPredictResult>>
//  ocr(cv::Mat srcimg, bool det = false, bool rec = true, bool cls = false);

private:
  shared_ptr<Yolo::Infer> engine{nullptr};
//  CRNNRecognizer *recognizer_ = nullptr;
};
}// namespace Yolov5Det
