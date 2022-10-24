#ifndef AUTODRONE_JETSON_MODEL
#define AUTODRONE_JETSON_MODEL

#include <yaml-cpp/yaml.h>

namespace jetson {

class Model {
 private:
  YAML::Node &config_;

 public:
 public:
  Model(YAML::Node &config);
  int Init();
  ~Model();
};

}  // namespace jetson
#endif  // AUTODRONE_JETSON_MODEL