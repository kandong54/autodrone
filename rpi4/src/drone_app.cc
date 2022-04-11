#include "drone_app.h"

#include <opencv2/core.hpp>

#include "log.h"
#include "camera.h"
#include "tflite.h"

namespace rpi4
{
  DroneApp::DroneApp(/* args */)
  {
    camera_ = std::make_unique<Camera>();
    tflite_ = std::make_unique<TFLite>();
  }

  DroneApp::~DroneApp()
  {
  }

  bool DroneApp::IsConnected()
  {
    bool camera = camera_->IsOpened();
    return camera && camera;
  }

  void DroneApp::Run()
  {
    cv::Mat frame;
    while (true)
    {
      SPDLOG_DEBUG("Loop start");
      if (!camera_->CaptureImage(frame))
      {
        SPDLOG_ERROR("Failed to capture image!");
        continue;
      }
      if (!tflite_->Inference(frame))
      {
        SPDLOG_ERROR("Failed to inference image!");
        continue;
      }
      SPDLOG_DEBUG("Loop end");
    }
  }

  void DroneApp::BuildAndStart()
  {
    if (!tflite_->Load())
    {
      SPDLOG_CRITICAL("Failed to Load TFlite!");
      return;
    }
    if (!camera_->Open())
    {
      SPDLOG_CRITICAL("Failed to open camera!");
      return;
    }
    if (!IsConnected())
    {
      SPDLOG_CRITICAL("Failed to init DroneApp!");
      return;
    }
    thread_ = std::make_unique<std::thread>([this] { Run(); });
    thread_->detach();
  }
} // namespace rpi4