#ifndef AUTODRONE_RPI4_SERVER
#define AUTODRONE_RPI4_SERVER

#include <grpcpp/grpcpp.h>

#include "drone.grpc.pb.h"

using autodrone::CameraReply;
using autodrone::Drone;
using autodrone::HelloReply;
using autodrone::HelloRequest;
using google::protobuf::Empty;
using grpc::Server;
using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::Status;

namespace rpi4
{
  class DroneApp;

  class DroneServiceImpl final : public Drone::Service
  {
  private:
    std::unique_ptr<Server> server_;
    DroneApp *drone_app_;
    std::string server_address_;

  public:
    DroneServiceImpl(DroneApp *drone_app);
    ~DroneServiceImpl();
    Status SayHello(ServerContext *context, const HelloRequest *request, HelloReply *reply) override;
    Status GetCamera(ServerContext *context, const Empty *request, ServerWriter<CameraReply> *writer) override;
    void Run();
    void Wait();
  };

} // namespace rpi4
#endif // AUTODRONE_RPI4_SERVER