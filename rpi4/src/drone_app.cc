#include "drone_app.h"

#include "log.h"

namespace rpi4
{
  DroneApp::DroneApp(/* args */)
  {
    camera = std::make_unique<Camera>();
    tflite = std::make_unique<TFLite>();
  }

  DroneApp::~DroneApp()
  {
  }

  bool DroneApp::IsConnected()
  {
    bool camera_flag = camera->IsOpened();
    bool tflite_flag = tflite->IsWork();
    return camera_flag && tflite_flag;
  }

  void DroneApp::Run()
  {
    std::unique_lock drone_1_lock(mutex_1);
    std::unique_lock drone_2_lock(mutex_2);
    mutex_1.lock();
    // mutex_2.lock();
    while (true)
    {
      SPDLOG_DEBUG("Loop start");
      if (!camera->CaptureImage(frame))
      {
        SPDLOG_ERROR("Failed to capture image!");
        continue;
      }
      mutex_2.lock();
      mutex_1.unlock();
      cv.notify_all();
      if (!tflite->Inference(frame))
      {
        SPDLOG_ERROR("Failed to inference image!");
        continue;
      }
      mutex_1.lock();
      mutex_2.unlock();
      cv.notify_all();
      SPDLOG_TRACE("Loop end");
    }
  }

  void DroneApp::BuildAndStart()
  {
    if (!tflite->Load())
    {
      SPDLOG_CRITICAL("Failed to Load TFlite!");
      return;
    }
    if (!camera->Open())
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