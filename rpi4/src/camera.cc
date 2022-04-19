#include "camera.h"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace rpi4
{

  Camera::Camera()
  {
    // init parameters
    cap_device_ = 0;
    cap_width_ = 960;
    cap_height_ = 720;
    cap_fps_ = 5;

    out_width = 640;
    out_height = 640;

    lock_ = std::unique_lock<std::mutex>(mutex, std::defer_lock);
  }

  bool Camera::Open()
  {
    SPDLOG_INFO("Opening camera {}", cap_device_);
    cap_.open(cap_device_);
    if (!cap_.isOpened())
    {
      SPDLOG_CRITICAL("Unable to open camera");
      return false;
    }
    // the setting must meet v4l2-ctl --list-formats-ext
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, cap_width_);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, cap_height_);
    cap_.set(cv::CAP_PROP_FPS, cap_fps_);
    // reduce buffer size so that we can read the most latest one
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    // warmup capture
    cap_.grab();
    SPDLOG_INFO("Opened camera {}", cap_device_);
    return true;
  }

  bool Camera::CaptureImage(cv::Mat &frame)
  {
    if (!cap_.isOpened())
      return false;
    // clear buffer
    // TODO: any good solution? How to read the most latest one?
    SPDLOG_TRACE("Read");
    // cap_.grab();
    cap_.read(frame);
    // check if we succeeded
    if (frame.empty())
    {
      SPDLOG_ERROR("blank frame grabbed");
      return false;
    }
    SPDLOG_TRACE("Postprocess");
    // TODO: add crop method.
    // cv::Rect crop((cap_width_ - out_width_) / 2, (cap_height_ - out_height_) / 2, (cap_width_ + out_width_) / 2, (cap_height_ + out_height_) / 2);
    // cv::Mat frame = frame(crop);
    cv::resize(frame, frame, cv::Size(out_width, out_height), 0, 0, cv::INTER_LINEAR);
    // cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    SPDLOG_TRACE("Finish");
    return true;
  }

  bool Camera::IsOpened()
  {
    return cap_.isOpened();
  }
} // namespace rpi4