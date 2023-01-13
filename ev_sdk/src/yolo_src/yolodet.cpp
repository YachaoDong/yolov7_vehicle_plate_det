#include "yolodet.h"
#include <numeric>
//#include <sys/stat.h>

namespace Yolov7Det {
//     static bool ifFileExists(const char *FileName)
//     {
//         struct stat my_stat;
//         return (stat(FileName, &my_stat) == 0);
//     }


YOLODET::YOLODET(){};

void YOLODET::DetInit(int deviceid, const string& engine_file, float confidence_threshold, float nms_threshold) {
    // auto engine = std::make_shared<Yolo::create_infer>(

    // TODO 使用 onnx转engine_file 进行推理测试，此处的engine_file为.onnx文件
    std::string model_file = engine_file;
    size_t sep_pos = model_file.find_last_of(".");
    model_file = model_file.substr(0, sep_pos) + ".trt";
    std::cout<<"model_file: "<< model_file << std::endl;

    if(not iLogger::exists(model_file))
    {
        std::cout<<"Convert the onnx model to trt model..."<< std::endl;
        // 将onnx转为trt模型
        TRT::compile(
            TRT::Mode::FP32,            // 使用fp32模型编译
            1,                          // max batch size
            engine_file,              // onnx 文件
            model_file,               // 保存的文件路径
            {}                         // 重新定制输入的shape
        );
    }



    this->engine = Yolo::create_infer(
    model_file,                // engine file
    Yolo::Type::V7,                       // yolo type, Yolo::Type::V5 / Yolo::Type::V7 / Yolo::Type type
    deviceid,                   // gpu id
    confidence_threshold,                      // confidence threshold 0.25f
    nms_threshold,                      // nms threshold  0.45f
    Yolo::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
    1024,                       // max objects
    false                       // preprocess use multi stream
    );
    
    if(this->engine == nullptr){
        std::cout << "Engine is nullptr" << std::endl;
        //INFOE("Engine is nullptr");
        //return;
    }
};
    
void YOLODET::DetUnInit(){
    if(this->engine.get() != nullptr){
        this->engine.reset();
    }
    
};


ObjectDetector::BoxArray YOLODET::commit_gpu(cv::Mat srcimg){

    if(this->engine == nullptr){
        std::cout << "Engine is nullptr!!!" << std::endl;
        }
    // 推理并获取结果
    auto det_result = this->engine->commit(srcimg).get();  // 得到的是vector<Box>

    return det_result;
}; // namespace PaddleOCR


YOLODET::~YOLODET() {
  DetUnInit();
  //this->engine.reset();
  // if (this->engine != nullptr) {
  //   delete this->engine;
  // }
};

} // namespace PaddleOCR
