#include "server.h"

#include <iostream>
#include <memory>
#include <string>

// #include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "drone.grpc.pb.h"

using autodrone::Drone;
using autodrone::HelloReply;
using autodrone::HelloRequest;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

namespace rpi4
{
  namespace
  {
    // Logic and data behind the server's behavior.
    class DroneServiceImpl final : public Drone::Service
    {
      Status SayHello(ServerContext *context, const HelloRequest *request,
                      HelloReply *reply) override
      {
        std::string prefix("Hello ");
        reply->set_message(prefix + request->name());
        return Status::OK;
      }
    };
  } // namespace
  void RunServer()
  {
    std::string server_address("0.0.0.0:9090");
    DroneServiceImpl service;

    grpc::EnableDefaultHealthCheckService(true);
    // grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
  }
} // AUTODRONE_RPI4_SERVER