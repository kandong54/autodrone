#include "camera.h"

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>

#include "config.h"

namespace rpi4
{

  Camera::Camera()
  {
    // TODO: remove default value
    cap_device_ = 0;
    cap_width = 960;
    cap_height = 720;
    cap_fps_ = 7.5;

    jpg_quality_ = 80;
    jpg_params_ = {cv::IMWRITE_JPEG_QUALITY, jpg_quality_};
  }
  void Camera::LoadConfig(Config *config)
  {
    if (config->node["camera"])
    {
      if (config->node["camera"]["device"])
      {
        cap_device_ = config->node["camera"]["device"].as<int>();
      }
      if (config->node["camera"]["width"])
      {
        cap_width = config->node["camera"]["width"].as<int>();
      }
      if (config->node["camera"]["height"])
      {
        cap_height = config->node["camera"]["height"].as<int>();
      }
      if (config->node["camera"]["fps"])
      {
        cap_fps_ = config->node["camera"]["fps"].as<double>();
      }
      if (config->node["camera"]["jpeg_quality"])
      {
        jpg_quality_ = config->node["camera"]["jpeg_quality"].as<int>();
        jpg_params_[1] = jpg_quality_;
      }
    }
  }
  void Camera::SetOutputSize(int width, int height)
  {
    out_width_ = width;
    out_height_ = height;
  }
  int Camera::Open()
  {
    int ret;
    SPDLOG_INFO("Open camera {}", cap_device_);
    // default GStreamer
    cap_.open(cap_device_, cv::CAP_V4L2);
    if (!cap_.isOpened())
    {
      SPDLOG_CRITICAL("Unable to open camera");
      return -1;
    }
    // the setting must meet v4l2-ctl --list-formats-ext
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, cap_width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, cap_height);
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
    return 0;
  }

  int Camera::Capture(cv::Mat &mat)
  {
    int ret;
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
    // TODO: add interpolation option
    cv::resize(mat_cap_, mat_resize_, cv::Size(out_width_, out_height_), 0, 0, cv::INTER_NEAREST);
    SPDLOG_TRACE("Encode image");
    encoded.clear();
    cv::imencode(".jpg", mat_resize_, encoded, jpg_params_);
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