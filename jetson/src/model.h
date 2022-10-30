#ifndef AUTODRONE_JETSON_MODEL
#define AUTODRONE_JETSON_MODEL

#include <jetson-inference/tensorNet.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/core.hpp>

namespace jetson {

class Model : public tensorNet {
 private:
  YAML::Node &config_;
  int model_size_;
  int camera_width_;
  int camera_height_;
  float conf_threshold_;
  float iou_threshold_;

 public:
  std::vector<cv::Rect> boxes;
  std::vector<float> confs;
  std::vector<int> class_id;
  std::vector<int> indices;

 public:
  Model(YAML::Node &config);
  int Init();
  void Process(void* input);
  void PostProcess();
  ~Model();
};

  // [Cx, Cy, width , height, confidence, class]
  enum OutputArray
  {
    kXCenter = 0,
    kYCenter = 1,
    kWidth = 2,
    kHeight = 3,
    kConfidence = 4,
    kClass = 5,
    kArrayLen = 6,
  };

}  // namespace jetson
#endif  // AUTODRONE_JETSON_MODEL