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
#include <spdlog/spdlog.h>

#include "drone_app.h"
#include "config.h"

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
      std::string Hash(const std::string &password, const std::string &salt)
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
        return hex_stream.str();
      }

      DroneAuthMetadataProcessor(const std::string &password_hashed)
      {
        token_ = std::string("Bearer ") + password_hashed;
      }

      // DroneAuthMetadataProcessor(const std::string &password, const std::string &salt)
      // {
      //   token_ = std::string("Bearer ") + Hash(password, salt);
      // }

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

  DroneServiceImpl::DroneServiceImpl(Config *config, DroneApp *drone_app)
  {
    config_ = config;
    drone_app_ = drone_app;
    // TODO: remove default value
    server_address_ = "0.0.0.0:9090";
    server_key_path_ = "./certs/key.pem";
    server_cert_path_ = "./certs/cert.pem";
    // password_ = "robobee";
    // password_salt_ = "3NqlrT9*v8^0";

    if (config->node["server"])
    {
      if (config->node["server"]["address"])
      {
        server_address_ = config->node["server"]["address"].as<std::string>();
      }
      if (config->node["server"]["key_path"])
      {
        server_key_path_ = config->node["server"]["key_path"].as<std::string>();
      }
      if (config->node["server"]["cert_path"])
      {
        server_cert_path_ = config->node["server"]["cert_path"].as<std::string>();
      }
      if (config->node["server"]["password"])
      {
        password_hashed_ = config->node["server"]["password"].as<std::string>();
      }
    }
    // processor_ = std::unique_ptr<DroneAuthMetadataProcessor>(new DroneAuthMetadataProcessor(password_, password_salt_));
    processor_ = std::unique_ptr<DroneAuthMetadataProcessor>(new DroneAuthMetadataProcessor(password_hashed_));
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
    // TODO: remove mutex
    std::mutex mutex;
    std::unique_lock drone_lock(mutex);
    CameraReply reply;
    // TODO: drone_app_->tflite
    // size_t image_bytes = drone_app_->frame.total() * drone_app_->frame.elemSize();
    while (!context->IsCancelled())
    {
      SPDLOG_DEBUG("Loop start");
      // Read Image
      drone_app_->cv.wait(drone_lock, [this]
                          { return drone_app_->cv_flag; });
      SPDLOG_TRACE("set image");
      reply.set_image(drone_app_->camera.encoded.data(), drone_app_->camera.encoded.size());
      // Bounding Box
      SPDLOG_TRACE("add box");
      reply.clear_box();
      for (int i : drone_app_->tflite.indices)
      {
        CameraReply_BoundingBox *box = reply.add_box();
        box->set_left(drone_app_->tflite.boxes[i].x);
        box->set_top(drone_app_->tflite.boxes[i].y);
        box->set_width(drone_app_->tflite.boxes[i].width);
        box->set_height(drone_app_->tflite.boxes[i].height);
        box->set_confidence(drone_app_->tflite.confs[i]);
        box->set_class_(drone_app_->tflite.class_id[i]);
      }
      SPDLOG_TRACE("send reply");
      writer->Write(reply);
      SPDLOG_TRACE("Loop end");
      // Avoid fake wake
      // TODO: better sulotion?
      drone_app_->cv_flag = false;
    }
    return Status::OK;
  }

  Status DroneServiceImpl::GetImageSize([[maybe_unused]] ServerContext *context, [[maybe_unused]] const Empty *request, ImageSize *reply)
  {
    SPDLOG_INFO("GetImageSize");
    reply->set_image_width(drone_app_->tflite.input_width);
    reply->set_image_height(drone_app_->tflite.input_height);
    reply->set_camera_width(drone_app_->camera.cap_width);
    reply->set_camera_height(drone_app_->camera.cap_height);
    return Status::OK;
  }
} // AUTODRONE_RPI4_SERVER