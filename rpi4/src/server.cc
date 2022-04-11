#include "server.h"

#include <mutex>
#include <condition_variable>

// #include <grpcpp/ext/proto_server_reflection_plugin.h>
// #include <grpcpp/health_check_service_interface.h>

#include "drone_app.h"
#include "log.h"

using autodrone::CameraReply_BoundingBox;
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

  Status DroneServiceImpl::SayHello(ServerContext *context, const HelloRequest *request, HelloReply *reply)
  {
    SPDLOG_INFO("SayHello: {}", request->name());
    std::string prefix("Hello ");
    reply->set_message(prefix + request->name());
    return Status::OK;
  }
  Status DroneServiceImpl::GetCamera(ServerContext *context, const Empty *request, ServerWriter<CameraReply> *writer)
  {
    SPDLOG_INFO("GetCamera");
    // std::unique_lock camera_lock(drone_app_->camera->mutex);
    // std::unique_lock tflite_lock(drone_app_->tflite->mutex);
    std::unique_lock drone_1_lock(drone_app_->mutex_1);
    std::unique_lock drone_2_lock(drone_app_->mutex_2);
    CameraReply reply;
    drone_app_->cv.wait(drone_1_lock);
    drone_1_lock.unlock();
    size_t image_bytes = drone_app_->frame.total() * drone_app_->frame.elemSize();
    while (!context->IsCancelled())
    {
      // Read Image
      drone_app_->cv.wait(drone_1_lock);
      drone_1_lock.unlock();
      reply.set_image(drone_app_->frame.ptr(), image_bytes);
      // Bounding Box
      drone_app_->cv.wait(drone_2_lock);
      drone_2_lock.unlock();
      reply.clear_box();
      size_t num_box = drone_app_->tflite->prediction.size();
      for (size_t i = 0; i < num_box; i++)
      {
        CameraReply_BoundingBox *box = reply.add_box();
        box->set_x_min(drone_app_->tflite->prediction[i * kOutputNum + kXmin]);
        box->set_y_min(drone_app_->tflite->prediction[i * kOutputNum + kYmin]);
        box->set_x_max(drone_app_->tflite->prediction[i * kOutputNum + kXmax]);
        box->set_y_max(drone_app_->tflite->prediction[i * kOutputNum + kYmax]);
        box->set_confidence(drone_app_->tflite->prediction[i * kOutputNum + kConfidence]);
        box->set_class_(drone_app_->tflite->prediction[i * kOutputNum + kClass]);
      }
      writer->Write(reply);
    }
    return Status::OK;
  }
} // AUTODRONE_RPI4_SERVER