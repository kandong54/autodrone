
#include "log.h"
#include "drone_app.h"
#include "server.h"

int main(int argc, char *argv[])
{
  spdlog::set_level(spdlog::level::trace);
  // [2022-04-10 13:41:10.003] [Elapsed time] [info] [main.cc:main():9] [Thread id]  Hello World
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%4o] [%^%l%$] [%s:%!():%#] [%t] %v");
  SPDLOG_INFO("Hello World");

  // TODO: config

  // Drone
  rpi4::DroneApp drone;
  drone.BuildAndStart();

  // gRPC server
  rpi4::DroneServiceImpl server(&drone);
  server.Run();

  // Wait
  server.Wait();

  return 0;
}