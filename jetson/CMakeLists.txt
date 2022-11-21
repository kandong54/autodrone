cmake_minimum_required(VERSION 3.16)
project(drone VERSION 0.1.0)

# pass version to the source code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Dependencies
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(jetson-utils REQUIRED)
find_package(jetson-inference REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(vpi REQUIRED)
include(cmake/nvmpi.cmake)

# Protos
add_subdirectory(protos)

add_executable(${CMAKE_PROJECT_NAME} 
  src/main.cc
  src/server.cc
  src/camera.cc
  src/model.cc
)

# spdlog
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO
)

target_include_directories(drone PRIVATE
  src
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

target_link_directories(drone PRIVATE
  /usr/lib/aarch64-linux-gnu/tegra
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
  spdlog::spdlog
  yaml-cpp
  grpc_proto
  OpenSSL::Crypto
  stdc++fs
  jetson-utils
  jetson-inference
  nvmpi
  vpi
  ${OpenCV_LIBS}
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
)
