#include "config.h"

#include <fstream>
#include <filesystem>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

namespace rpi4
{
  Config::Config()
  {
  }

  Config::~Config()
  {
  }

  int Config::Load(const std::string &filename)
  {
    SPDLOG_INFO("Load config {}", filename);
    if (!std::filesystem::exists(filename))
    {
      SPDLOG_CRITICAL("{} does not exist", filename);
      return -1;
    }
    filename_ = filename;
    node = YAML::LoadFile(filename);
    return 0;
  }

  void Config::Save()
  {
    std::ofstream fout(filename_);
    fout << node;
  }

} // namespace rpi4