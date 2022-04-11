#include "server.h"

#include <iostream>
#include <memory>
#include <string>

// #include <grpcpp/ext/proto_server_reflection_plugin.h>
// #include <grpcpp/health_check_service_interface.h>

#include "drone_app.h"
#include "log.h"

using grpc::ServerBuilder;

namespace rpi4
{
  DroneServiceImpl::DroneServiceImpl(DroneApp *drone_app)
  {
    drone_app_ = drone_app;
    server_address_ = "0.0.0.0:9090";
  }

  DroneServiceImpl::~DroneServiceImpl()
  {
  }

  void DroneServiceImpl::Run()
  {
    // grpc::EnableDefaultHealthCheckService(true);
    // grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(this);
    // Finally assemble the server.
    server_ = builder.BuildAndStart();
    SPDLOG_WARN("Server listening on {}", server_address_);
  }

  void DroneServiceImpl::Wait()
  {
    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server_->Wait();
  }

  Status DroneServiceImpl::SayHello(ServerContext *context, const HelloRequest *request,
                                    HelloReply *reply)
  {
    SPDLOG_INFO("SayHello: {}", request->name());
    std::string prefix("Hello ");
    reply->set_message(prefix + request->name());
    return Status::OK;
  }

} // AUTODRONE_RPI4_SERVER