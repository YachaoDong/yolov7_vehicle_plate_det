// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

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

#include "ocr_cls.h"
#include "ocr_det.h"
#include "ocr_rec.h"
#include "preprocess_op.h"
#include "utility.h"

using namespace paddle_infer;

namespace PaddleOCR {

class PPOCR {
public:
  PPOCR();
  ~PPOCR();
  void OcrInit(const std::string &model_dir = "/project/train/models/34895/rec/mv3_none_bilstm_ctc/inference",
               const bool &use_gpu=true,
               const int &gpu_id=0,
               const int &gpu_mem=4000,
               const int &cpu_math_library_num_threads=8,
               const bool &use_mkldnn=true,
               const string &rec_char_dict_path="/usr/local/ev_sdk/src/paddle_src/src//plate_dict.txt",
               const bool &use_tensorrt=false,
               const std::string &precision="fp32",    // "Precision be one of fp32/fp16/int8"
               const int &rec_batch_num=1,
               const int &rec_img_h=32,
               const int &rec_img_w=100);
    
  void OcrUnInit();
    
  std::vector<std::vector<OCRPredictResult>>
  ocr(cv::Mat srcimg, bool det = false, bool rec = true, bool cls = false);

private:
  DBDetector *detector_ = nullptr;
  Classifier *classifier_ = nullptr;
  CRNNRecognizer *recognizer_ = nullptr;

  void det(cv::Mat img, std::vector<OCRPredictResult> &ocr_results,
           std::vector<double> &times);
  void rec(std::vector<cv::Mat> img_list,
           std::vector<OCRPredictResult> &ocr_results,
           std::vector<double> &times);
  void cls(std::vector<cv::Mat> img_list,
           std::vector<OCRPredictResult> &ocr_results,
           std::vector<double> &times);
  void log(std::vector<double> &det_times, std::vector<double> &rec_times,
           std::vector<double> &cls_times, int img_num);
};

} // namespace PaddleOCR
