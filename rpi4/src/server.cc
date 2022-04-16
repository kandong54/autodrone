#include "server.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <mutex>
#include <condition_variable>

// #include <grpcpp/ext/proto_server_reflection_plugin.h>
// #include <grpcpp/health_check_service_interface.h>
#include <openssl/evp.h>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include "drone_app.h"

using autodrone::CameraReply_BoundingBox;
using grpc::AuthContext;
using grpc::ServerBuilder;
using grpc::StatusCode;

namespace rpi4
{
  namespace
  {
    // https://github.com/grpc/grpc/blob/master/test/cpp/util/test_credentials_provider.cc
    std::string ReadFile(const std::string &src_path)
    {
      std::ifstream src;
      src.open(src_path, std::ifstream::in | std::ifstream::binary);

      std::string contents;
      src.seekg(0, std::ios::end);
      contents.reserve(src.tellg());
      src.seekg(0, std::ios::beg);
      contents.assign((std::istreambuf_iterator<char>(src)),
                      (std::istreambuf_iterator<char>()));
      return contents;
    }
    // https://github.com/grpc/grpc/blob/master/test/cpp/end2end/end2end_test.cc
    class DroneAuthMetadataProcessor : public AuthMetadataProcessor
    {
    public:
      DroneAuthMetadataProcessor(const std::string &password, const std::string &salt)
      {
        std::string to_be_hashed = password + salt;

        const EVP_MD *md_alg = EVP_sha256();
        unsigned int md_len = EVP_MD_size(md_alg);
        unsigned char md_value[md_len];

        EVP_Digest(to_be_hashed.c_str(), to_be_hashed.size(),
                   md_value, &md_len,
                   md_alg, nullptr);
        std::ostringstream hex_stream;

        for (unsigned int i = 0; i < md_len; i++)
          hex_stream << std::hex << std::setfill('0') << std::setw(2) << (int)md_value[i];

        token_ = std::string("Bearer ") + hex_stream.str();
      }

      // Interface implementation
      bool IsBlocking() const override { return false; }

      Status Process(const InputMetadata &auth_metadata,
                     [[maybe_unused]] AuthContext *context,
                     OutputMetadata *consumed_auth_metadata,
                     [[maybe_unused]] OutputMetadata *response_metadata) override
      {
        auto auth_md = auth_metadata.find("authorization");
        grpc::string_ref auth_md_value = auth_md->second;
        if (auth_md_value == token_)
        {
          // context->AddProperty(kIdentityPropName, kGoodGuy);
          // context->SetPeerIdentityPropertyName(kIdentityPropName);
          consumed_auth_metadata->insert(std::make_pair(
              std::string(auth_md->first.data(), auth_md->first.length()),
              std::string(auth_md->second.data(), auth_md->second.length())));
          return Status::OK;
        }
        else
        {
          return Status(StatusCode::UNAUTHENTICATED, std::string("Invalid password!"));
        }
      }

    private:
      std::string token_;
    };
  } // namespace

  DroneServiceImpl::DroneServiceImpl(DroneApp *drone_app)
  {
    drone_app_ = drone_app;
    server_address_ = "0.0.0.0:9090";
    server_key_path_ = "./certs/key.pem";
    server_cert_path_ = "./certs/cert.pem";
    password_ = "robobee";
    password_salt_ = "3NqlrT9*v8^0";
    processor_ = std::unique_ptr<DroneAuthMetadataProcessor>(new DroneAuthMetadataProcessor(password_, password_salt_));
  }

  DroneServiceImpl::~DroneServiceImpl()
  {
  }

  void DroneServiceImpl::Run()
  {
    // grpc::EnableDefaultHealthCheckService(true);
    // grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    // Authentication
    grpc::SslServerCredentialsOptions ssl_opts;
    ssl_opts.pem_root_certs = "";
    std::string server_key = ReadFile(server_key_path_);
    std::string server_cert = ReadFile(server_cert_path_);
    grpc::SslServerCredentialsOptions::PemKeyCertPair pkcp = {server_key, server_cert};
    ssl_opts.pem_key_cert_pairs.push_back(pkcp);
    auto server_creds = grpc::SslServerCredentials(ssl_opts);
    // AuthMetadataProcessor
    server_creds->SetAuthMetadataProcessor(std::move(processor_));
    // Listen on the given address with given authentication mechanism.
    builder.AddListeningPort(server_address_, server_creds);
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

  Status DroneServiceImpl::SayHello([[maybe_unused]] ServerContext *context, const HelloRequest *request, HelloReply *reply)
  {
    SPDLOG_INFO("SayHello: {}", request->name());
    std::string prefix("Hello ");
    reply->set_message(prefix + request->name());
    return Status::OK;
  }
  Status DroneServiceImpl::GetCamera(ServerContext *context, [[maybe_unused]] const Empty *request, ServerWriter<CameraReply> *writer)
  {
    SPDLOG_INFO("GetCamera");
    // std::unique_lock camera_lock(drone_app_->camera->mutex, std::defer_lock);
    // std::unique_lock tflite_lock(drone_app_->tflite->mutex, std::defer_lock);
    // TODO: remove mutex
    std::mutex mutex;
    std::unique_lock drone_lock(mutex);
    CameraReply reply;
    std::vector<uchar> buf;
    // TODO: drone_app_->tflite
    buf.reserve(640 * 640 * 3);
    // drone_app_->cv_1.wait(drone_lock, [this] { return drone_app_->cv_flag_1; });
    // size_t image_bytes = drone_app_->frame.total() * drone_app_->frame.elemSize();
    while (!context->IsCancelled())
    {
      SPDLOG_DEBUG("Loop start");
      // Read Image
      drone_app_->cv_1.wait(drone_lock, [this]
                            { return drone_app_->cv_flag_1; });
      SPDLOG_TRACE("set image");
      buf.clear();
      cv::imencode(".jpg", drone_app_->frame, buf);
      reply.set_image(buf.data(), buf.size());
      SPDLOG_TRACE("set");
      // Bounding Box
      drone_app_->cv_2.wait(drone_lock, [this]
                            { return drone_app_->cv_flag_2; });
      reply.clear_box();
      SPDLOG_TRACE("add box");
      size_t num_box = drone_app_->tflite->prediction.size();
      for (size_t i = 0; i < num_box; i++)
      {
        CameraReply_BoundingBox *box = reply.add_box();
        box->set_x_center(drone_app_->tflite->prediction[i * kOutputNum + kXCenter]);
        box->set_y_center(drone_app_->tflite->prediction[i * kOutputNum + kYCenter]);
        box->set_width(drone_app_->tflite->prediction[i * kOutputNum + kWidth]);
        box->set_height(drone_app_->tflite->prediction[i * kOutputNum + kHeight]);
        box->set_confidence(drone_app_->tflite->prediction[i * kOutputNum + kConfidence]);
        box->set_class_(drone_app_->tflite->prediction[i * kOutputNum + kClass]);
      }
      SPDLOG_TRACE("added");
      writer->Write(reply);
      SPDLOG_TRACE("Loop end");
    }
    return Status::OK;
  }
} // AUTODRONE_RPI4_SERVER