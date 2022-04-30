#ifndef AUTODRONE_RPI4_SERVER
#define AUTODRONE_RPI4_SERVER

#include <grpcpp/grpcpp.h>
#include <grpcpp/security/auth_metadata_processor.h>

#include "protos/drone.grpc.pb.h"

using autodrone::CameraReply;
using autodrone::Drone;
using autodrone::Empty;
using autodrone::HelloReply;
using autodrone::HelloRequest;
using autodrone::ImageSize;
using grpc::AuthMetadataProcessor;
using grpc::Server;
using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::Status;

namespace rpi4
{
  class DroneApp;
  class Config;

  class DroneServiceImpl final : public Drone::Service
  {
  private:
    std::unique_ptr<Server> server_;
    DroneApp *drone_app_;
    Config *config_;
    std::string server_address_;
    std::string server_key_path_;
    std::string server_cert_path_;
    // std::string password_;
    // std::string password_salt_;
    std::string password_hashed_;

    std::unique_ptr<AuthMetadataProcessor> processor_;

  public:
    DroneServiceImpl(Config *config, DroneApp *drone_app);
    ~DroneServiceImpl();
    Status SayHello(ServerContext *context, const HelloRequest *request, HelloReply *reply) override;
    Status GetCamera(ServerContext *context, const Empty *request, ServerWriter<CameraReply> *writer) override;
    Status GetImageSize(ServerContext *context, const Empty *request, ImageSize *reply) override;
    void Run();
    void Wait();
  };

} // namespace rpi4
#endif // AUTODRONE_RPI4_SERVER