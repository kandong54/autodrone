#include <server.h>
// #include <camera.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <experimental/filesystem>

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
  if (!std::experimental::filesystem::exists(config_path)) {
    SPDLOG_CRITICAL("Config does not exist: {}", config_path);
    return -1;
  }
  YAML::Node config = YAML::LoadFile(config_path);

  /*
   * camera
   */
  // jetson::Camera camera(config);

  /*
   * TensorRT
   */

  /*
   * gRPC
   */
  jetson::DroneServiceImpl server(config);
  server.Run();
  server.Wait();
  return 0;

  while (true) {
    SPDLOG_TRACE("*** Strat ***");
    // camera.Capture();
    SPDLOG_TRACE("*** End ***");
  }

  return 0;
}