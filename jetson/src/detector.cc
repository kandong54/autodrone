#include "detector.h"

#include <jetson-inference/tensorConvert.h>
#include <jetson-utils/imageIO.h>
#include <nvbuf_utils.h>
#include <spdlog/spdlog.h>

#include <opencv2/dnn/dnn.hpp>

namespace jetson {

Detector::Detector(YAML::Node& config, int input_fd) : tensorNet(), config_(config), rgb_fd_(input_fd) {
  // save the parameters
  camera_width_ = config_["camera"]["width"].as<int>();
  camera_height_ = config_["camera"]["height"].as<int>();
  model_size_ = config_["detector"]["size"].as<int>();
  conf_threshold_ = config_["detector"]["confidence_threshold"].as<float>();
  iou_threshold_ = config_["detector"]["iou_threshold"].as<float>();
}

int Detector::Init() {
  CreateStream();
  std::vector<std::string> input_blobs = {config_["detector"]["input_layer"].as<std::string>()};
  std::vector<std::string> output_blobs = {config_["detector"]["output_layer"].as<std::string>()};
  // load model
  if (!LoadEngine(config_["detector"]["model_path"].as<std::string>().c_str(), input_blobs, output_blobs)) {
    SPDLOG_CRITICAL("Failed to LoadEngine");
  }
}

void Detector::Process() {
  SPDLOG_TRACE("Strat");
  SPDLOG_TRACE("Read Fd");
  // get memory address from fd
  egl_image_ = NvEGLImageFromFd(NULL, rgb_fd_);
  if (egl_image_ == NULL)
    SPDLOG_CRITICAL("NvEGLImageFromFd");
  cudaEglFrame eglFrame;
  eglResource_ = NULL;
  if (CUDA_FAILED(cudaGraphicsEGLRegisterImage(&eglResource_, egl_image_,
                                               CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)))
    SPDLOG_CRITICAL("cudaGraphicsEGLRegisterImage");
  if (CUDA_FAILED(cudaGraphicsResourceGetMappedEglFrame(&eglFrame, eglResource_, 0, 0)))
    SPDLOG_CRITICAL("cuGraphicsResourceGetMappedEglFrame");
  if (eglFrame.frameType != cudaEglFrameTypePitch)
    SPDLOG_CRITICAL("{} != cudaEglFrameTypePitch", eglFrame.frameType);

  // preProcess
  // https://github.com/dusty-nv/jetson-inference/blob/master/c/detectNet.cpp
  // https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet
  if (cudaTensorNormRGB((void*)eglFrame.frame.pArray[0], IMAGE_RGBA8, model_size_, model_size_,
                        mInputs[0].CUDA, model_size_, model_size_,
                        make_float2(0.0f, 1.0f),
                        GetStream()))
    SPDLOG_CRITICAL("cudaTensorNormRGB");

  SPDLOG_TRACE("ProcessNetwork");
  // non-block process network
  if (!ProcessNetwork(false))
    SPDLOG_CRITICAL("Failed to ProcessNetwork");

  SPDLOG_TRACE("End");
}

void Detector::PostProcess() {
  SPDLOG_TRACE("cudaStreamSynchronize");
  // wait results
  cudaStreamSynchronize(GetStream());

  SPDLOG_TRACE("cleanup");
  if (CUDA_FAILED(cudaGraphicsUnregisterResource(eglResource_)))
    SPDLOG_CRITICAL("cudaGraphicsUnregisterResource");
  NvDestroyEGLImage(NULL, egl_image_);

  // https://github.com/itsnine/yolov5-onnxruntime/blob/master/src/detector.cpp
  SPDLOG_TRACE("Loop");
  // switch buffer index
  buffer_index = 1 - buffer_index;
  // clear data
  boxes[buffer_index].clear();
  confs[buffer_index].clear();
  indices[buffer_index].clear();
  class_id[buffer_index].clear();
  // cpu address of output
  float* output_ptr = mOutputs[0].CPU;
  int nums = mOutputs[0].dims.d[1];
  // load bounding boxes
  for (int i = 0; i < (int)kArrayLen * nums; i += (int)kArrayLen) {
    // [Cx, Cy, width , height, confidence, class1, class2, ...]
    if (output_ptr[i + kConfidence] >= conf_threshold_) {
      int center_x = output_ptr[i + kXCenter] / model_size_ * camera_width_;
      int center_y = output_ptr[i + kYCenter] / model_size_ * camera_height_;
      int width = output_ptr[i + kWidth] / model_size_ * camera_width_;
      int height = output_ptr[i + kHeight] / model_size_ * camera_height_;
      int left = center_x - width / 2;
      int top = center_y - height / 2;
      boxes[buffer_index].emplace_back(left, top, width, height);
      confs[buffer_index].emplace_back(output_ptr[i + kConfidence]);
      class_id[buffer_index].emplace_back(1);
    }
  }
  // NMS
  SPDLOG_TRACE("NMSBoxes");
  cv::dnn::NMSBoxes(boxes[buffer_index], confs[buffer_index], conf_threshold_, iou_threshold_, indices[buffer_index]);
  SPDLOG_TRACE("Total: {}", indices[buffer_index].size());
  SPDLOG_TRACE("End");
}

Detector::~Detector() {
}

}  // namespace jetson