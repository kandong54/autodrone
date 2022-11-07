#include "model.h"

#include <jetson-inference/tensorConvert.h>
#include <jetson-utils/imageIO.h>
#include <spdlog/spdlog.h>

#include <opencv2/dnn/dnn.hpp>
namespace jetson {

Model::Model(YAML::Node& config) : config_(config) {
  camera_width_ = config_["camera"]["width"].as<int>();
  camera_height_ = config_["camera"]["height"].as<int>();
  model_size_ = config_["detector"]["size"].as<int>();
  conf_threshold_ = config_["detector"]["confidence_threshold"].as<float>();
  iou_threshold_ = config_["detector"]["iou_threshold"].as<float>();
}

int Model::Init() {
  std::vector<std::string> input_blobs = {config_["detector"]["input_layer"].as<std::string>()};
  std::vector<std::string> output_blobs = {config_["detector"]["output_layer"].as<std::string>()};
  if (!LoadEngine(config_["detector"]["model_path"].as<std::string>().c_str(), input_blobs, output_blobs)) {
    SPDLOG_CRITICAL("Failed to LoadEngine");
  }
}

void Model::Process(void* input) {
  SPDLOG_DEBUG("Strat");
  // preProcess
  // https://github.com/dusty-nv/jetson-inference/blob/master/c/detectNet.cpp
  // https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet
  if (cudaTensorNormRGB(input, IMAGE_RGBA8, camera_width_ / 2, camera_height_,
                        mInputs[0].CUDA, model_size_, model_size_,
                        make_float2(0.0f, 1.0f),
                        GetStream()))
    SPDLOG_CRITICAL("cudaTensorNormRGB");

  SPDLOG_TRACE("ProcessNetwork");
  if (!ProcessNetwork(true))
    SPDLOG_CRITICAL("Failed to ProcessNetwork");

  SPDLOG_TRACE("PostProcess");
  PostProcess();

  SPDLOG_TRACE("End");
}

void Model::PostProcess() {
  // https://github.com/itsnine/yolov5-onnxruntime/blob/master/src/detector.cpp
  SPDLOG_TRACE("Loop");
  boxes.clear();
  confs.clear();
  indices.clear();
  float* output_ptr = mOutputs[0].CPU;
  int nums = mOutputs[0].dims.d[1];
  for (int i = 0; i < (int)kArrayLen * nums; i += (int)kArrayLen) {
    // [Cx, Cy, width , height, confidence, class1, class2, ...]
    if (output_ptr[i + kConfidence] >= conf_threshold_) {
      int center_x = output_ptr[i + kXCenter] / model_size_ * camera_width_;
      int center_y = output_ptr[i + kYCenter] / model_size_ * camera_height_;
      int width = output_ptr[i + kWidth] / model_size_ * camera_width_;
      int height =output_ptr[i + kHeight] / model_size_ * camera_height_;
      int left = center_x - width / 2;
      int top = center_y - height / 2;
      boxes.emplace_back(left, top, width, height);
      confs.emplace_back(output_ptr[i + kConfidence]);
      class_id.emplace_back(1);
    }
  }
  // NMS
  SPDLOG_TRACE("NMSBoxes");
  cv::dnn::NMSBoxes(boxes, confs, conf_threshold_, iou_threshold_, indices);
  SPDLOG_TRACE("Total: {}", indices.size());
  SPDLOG_TRACE("End");
}

Model::~Model() {
}

}  // namespace jetson