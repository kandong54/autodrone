#ifndef AUTODRONE_RPI4_CONFIG
#define AUTODRONE_RPI4_CONFIG

#include <yaml-cpp/yaml.h>

namespace rpi4
{
  class Config
  {
  private:
    std::string filename_;

  public:
    YAML::Node node;

  public:
    Config();
    ~Config();
    int Load(const std::string &filename);
    void Save();
  };

} // namespace rpi4
#endif // AUTODRONE_RPI4_CONFIG