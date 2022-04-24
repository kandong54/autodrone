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
    int ret;
    cv_flag = false;
    cv::Mat tmp_frame;
    while (true)
    {
      SPDLOG_DEBUG("Loop start");
      ret = camera->Capture(tmp_frame);
      if (ret)
      {
        SPDLOG_ERROR("Failed to capture image!");
        continue;
      }
      // compress image
      cv_flag = false;
      ret = tflite->Inference(tmp_frame);
      if (ret)
      {
        SPDLOG_ERROR("Failed to inference image!");
        continue;
      }
      cv_flag = true;
      cv.notify_all();
      SPDLOG_TRACE("Loop end");
    }
  }

  void DroneApp::BuildAndStart()
  {
    int ret;
    ret = tflite->Load();
    if (ret)
    {
      SPDLOG_CRITICAL("Failed to Load TFlite!");
      return;
    }
    ret = camera->Open();
    if (ret)
    {
      SPDLOG_CRITICAL("Failed to open camera!");
      return;
    }
    if (!IsConnected())
    {
      SPDLOG_CRITICAL("Failed to init DroneApp!");
      return;
    }
    thread_ = std::make_unique<std::thread>(
        [this]
        {
          Run();
        });
    thread_->detach();
  }
} // namespace rpi4