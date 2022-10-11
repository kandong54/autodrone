cmake_minimum_required(VERSION 3.16)
project(drone VERSION 0.1.0)

# pass version to the source code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Dependencies
include(cmake/Dependencies.cmake)

# Protos
include(cmake/Protos.cmake)

add_executable(${CMAKE_PROJECT_NAME} 
  src/main.cc
  # src/server.cc
  src/camera.cc
)

# spdlog
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE
)

target_include_directories(drone PRIVATE
  src
  ${CUDA_INCLUDE_DIRS}
)

target_link_directories(drone PRIVATE
  /usr/lib/aarch64-linux-gnu/tegra
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
  spdlog::spdlog
  yaml-cpp
  # grpc_proto
  # grpc++
  # libprotobuf
  # OpenSSL::Crypto
  stdc++fs
  jetson-utils
)
