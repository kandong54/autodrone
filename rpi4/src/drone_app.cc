#include "drone_app.h"

#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

namespace rpi4
{
  DroneApp::DroneApp(/* args */)
  {
    camera = std::make_unique<Camera>();
    tflite = std::make_unique<TFLite>();
    frame.reserve(camera->out_height * camera->out_width * 3);
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
    cv_flag = false;
    cv::Mat tmp_frame;
    while (true)
    {
      SPDLOG_DEBUG("Loop start");
      if (!camera->Capture(tmp_frame))
      {
        SPDLOG_ERROR("Failed to capture image!");
        continue;
      }
      // compress image
      cv_flag = false;
      SPDLOG_TRACE("compress image");
      frame.clear();
      cv::imencode(".jpg", tmp_frame, frame);
      if (!tflite->Inference(tmp_frame))
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