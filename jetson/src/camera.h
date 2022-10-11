#ifndef AUTODRONE_JETSON_CAMERA
#define AUTODRONE_JETSON_CAMERA

#include <yaml-cpp/yaml.h>

namespace jetson {

class Camera {
 private:
  YAML::Node &config_;

 public:

 public:
  Camera(YAML::Node &config);
  int Open();
  int Capture();
  int Encode();
  bool IsOpened();
  ~Camera();
};

}  // namespace jetson
#endif  // AUTODRONE_JETSON_CAMERA