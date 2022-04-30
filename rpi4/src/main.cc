
#include <spdlog/spdlog.h>

#include "config.h"
#include "drone_app.h"
#include "server.h"

int main(int argc, char *argv[])
{
  spdlog::set_level(static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));
  // [2022-04-10 13:41:10.003] [Elapsed time] [info] [main.cc:main():9] [Thread id]  Hello World
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%4o] [%^%l%$] [%12s:%10!():%4#] [%4t] %v");
  SPDLOG_INFO("Hello World");

  // TODO: use libboost-program-options to load filename
  std::string config_path("config.yaml");

  // Config
  rpi4::Config config;
  int ret = config.Load(config_path);
  if (ret)
  {
    SPDLOG_CRITICAL("Failed to Load config!");
    return -1;
  }
  // Drone
  rpi4::DroneApp drone;
  ret = drone.BuildAndStart(&config);
  if (ret)
  {
    SPDLOG_CRITICAL("Failed to start drone!");
    return -1;
  }
  // gRPC server
  rpi4::DroneServiceImpl server(&config, &drone);
  server.Run();

  // Wait
  server.Wait();

  return 0;
}