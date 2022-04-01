#include <iostream>

#include <spdlog/spdlog.h>

#include "camera.h"
#include "server.h"

int main(int argc, char *argv[])
{
  SPDLOG_INFO("Hello World");
  rpi4::RunServer();
}