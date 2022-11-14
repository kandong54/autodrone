#include <camera.h>
#include <model.h>
#include <server.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <condition_variable>
#include <experimental/filesystem>
#include <mutex>

int main(int argc, char* argv[]) {
  /*
   * spdlog
   */
  spdlog::set_level(static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));
  // [2022-04-10 13:41:10.003] [Elapsed time] [info] [main.cc:main():9] [Thread id]  Hello World
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%4o] [%^%l%$] [%12s:%10!():%4#] [%4t] %v");

  SPDLOG_WARN("Hello World");

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
   * GPU
   */
  cudaDeviceReset();

  /*
   * TensorRT
   */
  jetson::Model model(config);
  model.Init();

  /*
   * camera
   */
  jetson::Camera camera(config, model);
  camera.Open();

  /*
   * Concurrency
   */
  std::condition_variable cv;
  std::mutex cv_m_;

  /*
   * gRPC
   */
  jetson::DroneServiceImpl server(config, camera, model, cv_m_, cv);
  server.Run();
  // server.Wait();

  /*
   * Main Loop
   */

  while (true) {
    SPDLOG_TRACE("*** Strat ***");
    camera.RunParallel();
    // camera.Capture();
    // camera.Encode();
    // camera.Depth();
    // camera.Detect();
    SPDLOG_TRACE("notify_all");
    {
      std::lock_guard lk(cv_m_);
      server.ready = true;
      server.jpeg_index = camera.encode_index;
      server.box_index = model.buffer_index;
    }
    cv.notify_all();
    SPDLOG_TRACE("*** End ***");
  }

  return 0;
}