#ifndef AUTODRONE_JETSON_DEPTH
#define AUTODRONE_JETSON_DEPTH

#include <cuda_egl_interop.h>
#include <jetson-inference/tensorNet.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/core/cuda.hpp>

namespace jetson {

class Detector;

class Depth : public tensorNet {
 private:
  YAML::Node &config_;
  int detector_size_;
  int model_size_;
  int camera_width_;
  int camera_height_;
  int rgb_fd_;
  Detector *detector_;
  float depth_k_;
  float depth_b_; 
  int quality_;
  cv::cuda::GpuMat *map_f32_;
  cv::cuda::GpuMat *map_u8_;
  // to get the memory address of fd
  EGLImageKHR egl_image_;
  cudaGraphicsResource *eglResource_;

 public:
  // 2 buffers for server to read
  // avoid potential data race
  // the main thread and server thread will always write/read different buffer
  static const int buffer_num = 2;
  unsigned int buffer_index = 0;
  cv::Mat *map_u8[buffer_num];
  int depth_map_size;

 public:
  Depth(YAML::Node &config, int input_fd, Detector *detector);
  int Init();
  void Process();
  void PostProcess();
  ~Depth();
};

}  // namespace jetson
#endif  // AUTODRONE_JETSON_DEPTH