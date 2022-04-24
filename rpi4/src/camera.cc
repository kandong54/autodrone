#include "camera.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>

namespace rpi4
{

  Camera::Camera()
  {
    // init parameters
    cap_device_ = 0;
    cap_width_ = 960;
    cap_height_ = 720;
    cap_fps_ = 7.5;

    out_width = 640;
    out_height = 640;
  }

  int Camera::Open()
  {
    SPDLOG_INFO("Opening camera {}", cap_device_);
    // default GStreamer
    cap_.open(cap_device_, cv::CAP_V4L2);
    if (!cap_.isOpened())
    {
      SPDLOG_CRITICAL("Unable to open camera");
      return -1;
    }
    // the setting must meet v4l2-ctl --list-formats-ext
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, cap_width_);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, cap_height_);
    // // MJPG
    // cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap_.set(cv::CAP_PROP_FPS, cap_fps_);
    // reduce buffer size so that we can read the most latest one
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    // // read raw data
    // cap_.set(cv::CAP_PROP_CONVERT_RGB, false);
    // cap_.set(cv::CAP_PROP_FORMAT, -1);
    // warmup capture
    cap_.grab();
    SPDLOG_INFO("Opened");
    return 0;
  }

  int Camera::Capture(cv::Mat &mat)
  {
    if (!cap_.isOpened())
      return -1;
    // clear buffer
    // TODO: any good solution? How to read the most latest one?
    SPDLOG_TRACE("Read");
    // ap_.grab();
    cap_.read(mat_cap_);
    // check if we succeeded
    if (mat_cap_.empty())
    {
      SPDLOG_ERROR("blank frame grabbed");
      return -1;
    }
    SPDLOG_TRACE("Resize");
    // TODO: add crop method.
    // cv::Rect crop((cap_width_ - out_width_) / 2, (cap_height_ - out_height_) / 2, (cap_width_ + out_width_) / 2, (cap_height_ + out_height_) / 2);
    // cv::Mat frame = frame(crop);
    cv::resize(mat_cap_, mat_resize_, cv::Size(out_width, out_height), 0, 0, cv::INTER_NEAREST);
    SPDLOG_TRACE("Compress image");
    encoded.clear();
    cv::imencode(".jpg", mat_resize_, encoded);
    // TODO: move BGR2RGB
    SPDLOG_TRACE("Color");
    cv::cvtColor(mat_resize_, mat, cv::COLOR_BGR2RGB);
    SPDLOG_TRACE("Finish");
    return 0;
  }

  bool Camera::IsOpened()
  {
    return cap_.isOpened();
  }

} // namespace rpi4