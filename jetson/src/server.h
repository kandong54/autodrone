#ifndef AUTODRONE_JETSON_SERVER
#define AUTODRONE_JETSON_SERVER

#undef Status

#include <grpcpp/grpcpp.h>
#include <grpcpp/security/auth_metadata_processor.h>
#include <yaml-cpp/yaml.h>

#include <condition_variable>
#include <mutex>

#include "drone.grpc.pb.h"

using autodrone::CameraRequest;
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

class Camera;
class Detector;
class Depth;

class DroneServiceImpl final : public Drone::Service {
 private:
  YAML::Node &config_;
  Camera &camera_;
  Detector &detector_;
  Depth &depth_;
  std::unique_ptr<Server> server_;
  // auth
  std::unique_ptr<AuthMetadataProcessor> processor_;
  std::string password_hashed_;
  // sync
  std::condition_variable &cv_;
  std::mutex &cv_m_;

 public:
  bool ready = false;
  // buffer index
  // this class will read the data in other classes' buffer
  unsigned int jpeg_index = 0;
  unsigned int box_index = 0;
  unsigned int depth_index = 0;

 public:
  DroneServiceImpl(YAML::Node &config, Camera &camera, Detector &detector, Depth &depth, std::mutex &cv_m, std::condition_variable &cv);
  ~DroneServiceImpl();
  Status SayHello(ServerContext *context, const HelloRequest *request, HelloReply *reply) override;
  Status GetCamera(ServerContext *context, const CameraRequest *request, ServerWriter<CameraReply> *writer) override;
  Status GetBox(ServerContext *context, const Empty *request, CameraReply *reply) override;
  Status GetImageSize(ServerContext *context, const Empty *request, ImageSize *reply) override;
  void Run();
  void Wait();
};

}  // namespace jetson
#endif  // AUTODRONE_JETSON_SERVER