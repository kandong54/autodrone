#include <iostream>

#include <spdlog/spdlog.h>

#include "camera.h"
#include "server.h"

int main(int argc, char *argv[])
{
  spdlog::info("Hello World");
  rpi4::RunServer();
}