cmake_minimum_required(VERSION 3.16)
project(drone VERSION 0.1.0)

# pass version to the source code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# # CTest
# include(CTest)
# enable_testing()

# Dependencies
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
# include(FindOpenSSL)
find_package(OpenSSL REQUIRED)
find_package(yaml-cpp REQUIRED)
include(FindPkgConfig)
# find_package(gRPC CONFIG REQUIRED)
pkg_search_module(GRPCPP REQUIRED grpc++)
pkg_search_module(TFLITE REQUIRED tflite)

# Protos
add_subdirectory(protos)

add_executable(drone 
  src/main.cc
  src/drone_app.cc
  src/camera.cc
  src/server.cc
  src/tflite.cc
  src/config.cc
)

# spdlog
target_compile_definitions(drone PRIVATE
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE
)

target_include_directories(drone PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_BINARY_DIR}
)

target_link_libraries(drone PRIVATE
  proto
  ${OpenCV_LIBS}
  spdlog::spdlog
  OpenSSL::Crypto
  ${GRPCPP_LIBRARIES}
  ${TFLITE_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)

# # CPack
# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)

