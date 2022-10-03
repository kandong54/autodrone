cmake_minimum_required(VERSION 3.16)
project(drone VERSION 0.1.0)

# pass version to the source code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Dependencies
include(Dependencies.cmake)

# Protos
# add_subdirectory(protos)

add_executable(${CMAKE_PROJECT_NAME} 
  src/main.cc
)

# spdlog
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
  SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
  spdlog::spdlog
  yaml-cpp
  grpc++
)
