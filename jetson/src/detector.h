#ifndef AUTODRONE_JETSON_DETECTOR
#define AUTODRONE_JETSON_DETECTOR

#include <jetson-inference/tensorNet.h>
#include <yaml-cpp/yaml.h>
#include <cuda_egl_interop.h>

#include <opencv2/core.hpp>

namespace jetson {

class Detector : public tensorNet {
 private:
  YAML::Node &config_;
  int model_size_;
  int camera_width_;
  int camera_height_;
  float conf_threshold_;
  float iou_threshold_;
  int rgb_fd_;
  EGLImageKHR egl_image_;
  cudaGraphicsResource* eglResource_;

 public:
  static const int buffer_num = 2;
  unsigned int buffer_index = 0;
  std::vector<cv::Rect> boxes[buffer_num];
  std::vector<float> confs[buffer_num];
  std::vector<int> class_id[buffer_num];
  std::vector<int> indices[buffer_num];
  std::vector<float> depth[buffer_num];

 public:
  Detector(YAML::Node &config, int input_fd);
  int Init();
  void Process();
  void PostProcess();
  ~Detector();
};

// [Cx, Cy, width , height, confidence, class]
enum OutputArray {
  kXCenter = 0,
  kYCenter = 1,
  kWidth = 2,
  kHeight = 3,
  kConfidence = 4,
  kClass = 5,
  kArrayLen = 6,
};

}  // namespace jetson
#endif  // AUTODRONE_JETSON_DETECTOR