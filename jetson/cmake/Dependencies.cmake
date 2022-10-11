include(FetchContent)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.10.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.7.0
  GIT_SHALLOW TRUE
)
set(YAML_CPP_BUILD_TESTS OFF)
FetchContent_MakeAvailable(yaml-cpp)

FetchContent_Declare(
  grpc
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG v1.49.1
  GIT_SHALLOW TRUE
)
set(ABSL_PROPAGATE_CXX_STD ON)
FetchContent_MakeAvailable(grpc)

set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
if(CMAKE_CROSSCOMPILING)
  find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif()

include(FindOpenSSL)