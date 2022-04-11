#ifndef AUTODRONE_RPI4_SERVER
#define AUTODRONE_RPI4_SERVER

#include <grpcpp/grpcpp.h>
#include <grpcpp/security/auth_metadata_processor.h>

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
using grpc::AuthMetadataProcessor;

namespace rpi4
{
  class DroneApp;

  class DroneServiceImpl final : public Drone::Service
  {
  private:
    std::unique_ptr<Server> server_;
    DroneApp *drone_app_;
    std::string server_address_;
    std::string server_key_path_;
    std::string server_cert_path_;
    std::string password_;
    std::string password_salt_; 
    std::unique_ptr<AuthMetadataProcessor> processor_;

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