# https://github.com/grpc/grpc/blob/master/examples/cpp/helloworld/CMakeLists.txt

set(proto_path "${CMAKE_CURRENT_SOURCE_DIR}/protos")

file(GLOB protos "${proto_path}/*.proto")
foreach(proto ${protos})
  get_filename_component(proto_we ${proto} NAME_WE)
  set(proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/${proto_we}.pb.cc")
  set(proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${proto_we}.pb.h")
  set(grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/${proto_we}.grpc.pb.cc")
  set(grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${proto_we}.grpc.pb.h")
  add_custom_command(
    OUTPUT "${proto_srcs}" "${proto_hdrs}" "${grpc_srcs}" "${grpc_hdrs}"
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
         --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
         -I "${proto_path}"
         --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
         "${proto}"
    DEPENDS "${proto}")
endforeach()

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# grpc_proto
file(GLOB grpc_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/*.pb.cc")
file(GLOB grpc_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/*.pb.h")
add_library(grpc_proto
  ${grpc_proto_srcs}
  ${grpc_proto_hdrs})
target_link_libraries(grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})