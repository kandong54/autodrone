cmake_minimum_required(VERSION 3.21)
project(proto VERSION 0.1.0)

# pass version to the source code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# # CTest
# include(CTest)
# enable_testing()

find_package(Protobuf REQUIRED)

add_library(proto 
  drone.proto
)

# https://github.com/faaxm/exmpl-cmake-grpc/blob/master/proto/CMakeLists.txt
protobuf_generate(TARGET proto LANGUAGE cpp)
protobuf_generate(TARGET proto LANGUAGE grpc GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc PLUGIN "protoc-gen-grpc=/usr/bin/grpc_cpp_plugin")

target_include_directories(proto
  INTERFACE ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(proto PRIVATE
  protobuf::libprotobuf
)


# # CPack
# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)

