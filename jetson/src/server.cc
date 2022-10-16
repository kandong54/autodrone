#include "server.h"

#include <condition_variable>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>

#include "camera.h"

// #include <grpcpp/ext/proto_server_reflection_plugin.h>
// #include <grpcpp/health_check_service_interface.h>
#include <openssl/evp.h>
#include <spdlog/spdlog.h>

using autodrone::CameraReply_BoundingBox;

namespace jetson {
namespace {
// https://github.com/grpc/grpc/blob/master/test/cpp/util/test_credentials_provider.cc
std::string ReadFile(const std::string &src_path) {
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
class DroneAuthMetadataProcessor : public AuthMetadataProcessor {
 public:
  std::string Hash(const std::string &password, const std::string &salt) {
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

  DroneAuthMetadataProcessor(const std::string &password_hashed) {
    token_ = std::string("Bearer ") + password_hashed;
  }

  // Interface implementation
  bool IsBlocking() const override { return false; }

  Status Process(const InputMetadata &auth_metadata,
                 [[maybe_unused]] grpc::AuthContext *context,
                 OutputMetadata *consumed_auth_metadata,
                 [[maybe_unused]] OutputMetadata *response_metadata) override {
    auto auth_md = auth_metadata.find("authorization");
    grpc::string_ref auth_md_value = auth_md->second;
    if (auth_md_value == token_) {
      // context->AddProperty(kIdentityPropName, kGoodGuy);
      // context->SetPeerIdentityPropertyName(kIdentityPropName);
      consumed_auth_metadata->insert(std::make_pair(
          std::string(auth_md->first.data(), auth_md->first.length()),
          std::string(auth_md->second.data(), auth_md->second.length())));
      return Status::OK;
    } else {
      return Status(grpc::StatusCode::UNAUTHENTICATED, std::string("Invalid password!"));
    }
  }

 private:
  std::string token_;
};
}  // namespace

DroneServiceImpl::DroneServiceImpl(YAML::Node &config, Camera &camera, std::mutex &cv_m, std::condition_variable &cv) : config_(config), camera_(camera), cv_m_(cv_m), cv_(cv) {
  std::string server_key_path = config_["server"]["key_path"].as<std::string>();
  std::string server_cert_path = config_["server"]["cert_path"].as<std::string>();
  if (!std::experimental::filesystem::exists(server_key_path)) {
    SPDLOG_CRITICAL("Server key does not exist: {}", server_key_path);
  }
  if (!std::experimental::filesystem::exists(server_cert_path)) {
    SPDLOG_CRITICAL("Server cert does not exist: {}", server_cert_path);
  }

  password_hashed_ = config_["server"]["password"].as<std::string>();
  processor_ = std::unique_ptr<DroneAuthMetadataProcessor>(new DroneAuthMetadataProcessor(password_hashed_));
}

DroneServiceImpl::~DroneServiceImpl() {
}

void DroneServiceImpl::Run() {
  // grpc::EnableDefaultHealthCheckService(true);
  // grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;
  // Authentication
  grpc::SslServerCredentialsOptions ssl_opts;
  ssl_opts.pem_root_certs = "";
  std::string server_key = ReadFile(config_["server"]["key_path"].as<std::string>());
  std::string server_cert = ReadFile(config_["server"]["cert_path"].as<std::string>());
  grpc::SslServerCredentialsOptions::PemKeyCertPair pkcp = {server_key, server_cert};
  ssl_opts.pem_key_cert_pairs.push_back(pkcp);
  auto server_creds = grpc::SslServerCredentials(ssl_opts);
  // AuthMetadataProcessor
  server_creds->SetAuthMetadataProcessor(std::move(processor_));
  // Listen on the given address with given authentication mechanism.
  builder.AddListeningPort(config_["server"]["address"].as<std::string>(), server_creds);
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(this);
  // Finally assemble the server.
  server_ = builder.BuildAndStart();
  SPDLOG_WARN("Server listening on {}", config_["server"]["address"].as<std::string>());
}

void DroneServiceImpl::Wait() {
  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server_->Wait();
}

Status DroneServiceImpl::SayHello([[maybe_unused]] ServerContext *context, const HelloRequest *request, HelloReply *reply) {
  SPDLOG_INFO("SayHello: {}", request->name());
  std::string prefix("Hello ");
  reply->set_message(prefix + request->name());
  return Status::OK;
}

Status DroneServiceImpl::GetCamera(ServerContext *context, [[maybe_unused]] const Empty *request, ServerWriter<CameraReply> *writer) {
  SPDLOG_INFO("GetCamera");
  CameraReply reply;
  while (!context->IsCancelled()) {
    // Read Image
    SPDLOG_TRACE("condition_variable");
    std::unique_lock lk(cv_m_);
    cv_.wait(lk, [this] { return ready; });
    SPDLOG_DEBUG("Strat");
    SPDLOG_TRACE("set image");
    reply.set_image(camera_.mjpeg_buffer[mjpeg_index].start, mjpeg_size);
    // Bounding Box
    SPDLOG_TRACE("add box");
    reply.clear_box();
    // for (int i : drone_app_->tflite.indices) {
    //   CameraReply_BoundingBox *box = reply.add_box();
    //   box->set_left(drone_app_->tflite.boxes[i].x);
    //   box->set_top(drone_app_->tflite.boxes[i].y);
    //   box->set_width(drone_app_->tflite.boxes[i].width);
    //   box->set_height(drone_app_->tflite.boxes[i].height);
    //   box->set_confidence(drone_app_->tflite.confs[i]);
    //   box->set_class_(drone_app_->tflite.class_id[i]);
    // }
    SPDLOG_TRACE("send reply");
    writer->Write(reply);
    SPDLOG_TRACE("Loop end");
    ready = false;
  }
  return Status::OK;
}

Status DroneServiceImpl::GetImageSize([[maybe_unused]] ServerContext *context, [[maybe_unused]] const Empty *request, ImageSize *reply) {
  SPDLOG_INFO("GetImageSize");
  reply->set_width(config_["camera"]["width"].as<unsigned int>());
  reply->set_height(config_["camera"]["height"].as<unsigned int>());
  return Status::OK;
}
}  // namespace jetson