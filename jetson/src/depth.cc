#include "depth.h"

#include <jetson-inference/tensorConvert.h>
#include <jetson-utils/imageIO.h>
#include <nvbuf_utils.h>
#include <spdlog/spdlog.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include "detector.h"

namespace jetson {

Depth::Depth(YAML::Node& config, int input_fd, Detector* detector) : tensorNet(), config_(config), rgb_fd_(input_fd), detector_(detector) {
  // save the parameters
  camera_width_ = config_["camera"]["width"].as<int>();
  camera_height_ = config_["camera"]["height"].as<int>();
  model_size_ = config_["depth"]["size"].as<int>();
  detector_size_ = config_["detector"]["size"].as<int>();
  depth_k_ = config_["depth"]["depth_k"].as<float>();
  depth_b_ = config_["depth"]["depth_b"].as<float>();
  quality_ = config_["depth"]["quality"].as<int>();
  depth_map_size = model_size_ * model_size_ ;
}

int Depth::Init() {
  CreateStream();
  std::vector<std::string> input_blobs = {config_["depth"]["input_layer"].as<std::string>()};
  std::vector<std::string> output_blobs = {config_["depth"]["output_layer"].as<std::string>()};
  // load model
  if (!LoadEngine(config_["depth"]["model_path"].as<std::string>().c_str(), input_blobs, output_blobs)) {
    SPDLOG_CRITICAL("Failed to LoadEngine");
  }
  // gpu map
  map_f32_ = new cv::cuda::GpuMat(model_size_, model_size_, CV_32FC1, mOutputs[0].CUDA);
  map_u8_ = new cv::cuda::GpuMat(model_size_, model_size_, CV_8UC1);
  // cpu map (buffer for server)
  map_u8[0] = new cv::Mat(model_size_, model_size_, CV_8UC1);
  map_u8[1] = new cv::Mat(model_size_, model_size_, CV_8UC1);
}

void Depth::Process() {
  SPDLOG_TRACE("Strat");
  // preProcess
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
  if (cudaTensorNormMeanRGB((void*)eglFrame.frame.pArray[0], IMAGE_RGBA8, model_size_, model_size_,
                            mInputs[0].CUDA, model_size_, model_size_,
                            make_float2(0.0f, 1.0f), make_float3(0.485f, 0.456f, 0.406f), make_float3(0.229f, 0.224f, 0.225f),
                            GetStream()))
    SPDLOG_CRITICAL("cudaTensorNormMeanRGB");

  SPDLOG_TRACE("ProcessNetwork");
  // non-block process network
  if (!ProcessNetwork(false))
    SPDLOG_CRITICAL("Failed to ProcessNetwork");

  SPDLOG_TRACE("End");
}

void Depth::PostProcess() {
  SPDLOG_TRACE("cudaStreamSynchronize");
  // wait results
  cudaStreamSynchronize(GetStream());

  SPDLOG_TRACE("cleanup");
  if (CUDA_FAILED(cudaGraphicsUnregisterResource(eglResource_)))
    SPDLOG_CRITICAL("cudaGraphicsUnregisterResource");
  NvDestroyEGLImage(NULL, egl_image_);

  SPDLOG_TRACE("depth Map");
  // switch buffer index
  buffer_index = 1 - buffer_index;
  double map_min, map_max;
  // get min and max in map
  cv::cuda::minMax(*map_f32_, &map_min, &map_max);
  // convert map into uint8 on GPU
  // this conversion is for server to reduce the data size
  // float is too large, so I need to convert the depth into uint8 first
  map_f32_->convertTo(*map_u8_, CV_8UC1, 255.0 / (map_max - map_min), -255.0 * map_min / (map_max - map_min));
  // copy to cpu
  map_u8_->download(*map_u8[buffer_index]);

  SPDLOG_TRACE("Loop");
  const int box_index = detector_->buffer_index;
  // clear data
  detector_->depth[box_index].clear();
  for (int i : detector_->indices[box_index]) {
    float depth = 0;
    const int x = detector_->boxes[box_index][i].x * model_size_ / camera_width_;
    const int y = detector_->boxes[box_index][i].y * model_size_ / camera_height_;
    const int width = detector_->boxes[box_index][i].width * model_size_ / camera_width_;
    const int height = detector_->boxes[box_index][i].height * model_size_ / camera_height_;
    // calculate average depth of the bounding box using gpu
    cv::cuda::GpuMat box(*map_f32_, cv::Rect(x, y, width, height));
    depth = cv::cuda::sum(box)[0] / (width * height);
    depth = depth_k_ * depth + depth_b_;
    // save the results
    detector_->depth[box_index].emplace_back(depth);
  }
  SPDLOG_TRACE("End");
}

Depth::~Depth() {
  delete map_f32_;
  delete map_u8[0];
  delete map_u8[1];
}

}  // namespace jetson