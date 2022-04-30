#include "drone_app.h"

#include <spdlog/spdlog.h>

namespace rpi4
{
  DroneApp::DroneApp()
  {
  }

  DroneApp::~DroneApp()
  {
  }

  bool DroneApp::IsConnected()
  {
    bool camera_flag = camera.IsOpened();
    bool tflite_flag = tflite.IsWork();
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
      ret = camera.Capture(tmp_frame);
      if (ret)
      {
        SPDLOG_ERROR("Failed to capture image!");
        continue;
      }
      // compress image
      cv_flag = false;
      ret = tflite.Inference(tmp_frame);
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

  int DroneApp::BuildAndStart(Config *config)
  {
    int ret;
    tflite.LoadConfig(config);
    ret = tflite.Load();
    if (ret)
    {
      SPDLOG_CRITICAL("Failed to Load TFlite!");
      return -1;
    }
    camera.LoadConfig(config);
    ret = camera.Open();
    if (ret)
    {
      SPDLOG_CRITICAL("Failed to open camera!");
      return -1;
    }
    camera.SetOutputSize(tflite.input_width, tflite.input_height);
    tflite.SetCameraSize(camera.cap_width, camera.cap_height);
    if (!IsConnected())
    {
      SPDLOG_CRITICAL("Failed to init DroneApp!");
      return -1;
    }
    thread_ = std::thread(
        [this]
        {
          Run();
        });
    thread_.detach();
    return 0;
  }
} // namespace rpi4