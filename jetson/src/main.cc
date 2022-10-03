#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

int main(int argc, char* argv[]) {
  /*
   * spdlog
   */
  spdlog::set_level(static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));
  // [2022-04-10 13:41:10.003] [Elapsed time] [info] [main.cc:main():9] [Thread id]  Hello World
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%4o] [%^%l%$] [%12s:%10!():%4#] [%4t] %v");

  SPDLOG_INFO("Hello World");

  /*
   * config
   */
  std::string config_path = (argc > 1) ? argv[1] : "config.yaml";
  YAML::Node config = YAML::LoadFile(config_path);

  return 0;
}