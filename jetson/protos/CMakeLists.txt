# https://github.com/faaxm/exmpl-cmake-grpc/blob/master/proto/CMakeLists.txt

find_package(protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

add_library(grpc_proto drone.proto)

target_link_libraries(grpc_proto PUBLIC
  protobuf::libprotobuf
  gRPC::grpc++
  # gRPC::grpc++_reflection
)
target_include_directories(grpc_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
protobuf_generate(TARGET grpc_proto LANGUAGE cpp)
protobuf_generate(TARGET grpc_proto LANGUAGE grpc GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}")
