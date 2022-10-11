#ifndef AUTODRONE_JETSON_SERVER
#define AUTODRONE_JETSON_SERVER

#include <grpcpp/grpcpp.h>
#include <grpcpp/security/auth_metadata_processor.h>
#include <yaml-cpp/yaml.h>

#include "drone.grpc.pb.h"

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

namespace jetson {

class DroneServiceImpl final : public Drone::Service {
 private:
  YAML::Node &config_;
  std::unique_ptr<Server> server_;
  std::unique_ptr<AuthMetadataProcessor> processor_;
  std::string server_address_;
  std::string server_key_path_;
  std::string server_cert_path_;
  std::string password_hashed_;

 public:
  DroneServiceImpl(YAML::Node &config);
  ~DroneServiceImpl();
  Status SayHello(ServerContext *context, const HelloRequest *request, HelloReply *reply) override;
  Status GetCamera(ServerContext *context, const Empty *request, ServerWriter<CameraReply> *writer) override;
  Status GetImageSize(ServerContext *context, const Empty *request, ImageSize *reply) override;
  void Run();
  void Wait();
};

}  // namespace jetson
#endif  // AUTODRONE_JETSON_SERVER