#include "drone_app.h"

#include <spdlog/spdlog.h>

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
    cv_flag_1 = false;
    cv_flag_2 = true;
    cv::Mat tmp_frame;
    while (true)
    {
      SPDLOG_DEBUG("Loop start");
      if (!camera->CaptureImage(tmp_frame))
      {
        SPDLOG_ERROR("Failed to capture image!");
        continue;
      }
      tmp_frame.copyTo(frame);
      cv_flag_1 = true;
      cv_flag_2 = false;
      cv_1.notify_all();
      if (!tflite->Inference(tmp_frame))
      {
        SPDLOG_ERROR("Failed to inference image!");
        continue;
      }
      cv_flag_1 = false;
      cv_flag_2 = true;
      cv_2.notify_all();
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
    thread_ = std::make_unique<std::thread>([this]
                                            { Run(); });
    thread_->detach();
  }
} // namespace rpi4