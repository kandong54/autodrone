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

    bit_rate_ = 10000000;
    bit_rate_ = bit_rate_ * video_fps_ / cap_fps_;
  }

  Camera::~Camera()
  {
    DelEncoder();
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
    SPDLOG_INFO("Init encoder");
    DelEncoder();
    ret = InitEncoder();
    if (ret < 0)
    {
      SPDLOG_CRITICAL("Failed to init encoder.");
      return -1;
    }
    return 0;
  }
  int Camera::InitEncoder()
  {
    // https://git.ffmpeg.org/gitweb/ffmpeg.git/blob/HEAD:/doc/examples/encode_video.c
    SPDLOG_INFO("Init Encoder");
    const char *codec_name = "h264_v4l2m2m";
    const AVCodec *codec;
    int ret;
    video_time_ = 0;
    /* find the mpeg1video encoder */
    codec = avcodec_find_encoder_by_name(codec_name);
    if (!codec)
    {
      SPDLOG_CRITICAL("Codec {} not found", codec_name);
      return -1;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_)
    {
      SPDLOG_CRITICAL("Could not allocate video codec context");
      return -1;
    }

    packet = av_packet_alloc();
    if (!packet)
    {
      SPDLOG_CRITICAL("Could not allocate packet");
      return -1;
    }

    /* put sample parameters */
    // v4l2-ctl --list-ctrls-menu -d 11
    codec_ctx_->bit_rate = bit_rate_;
    codec_ctx_->width = out_width;
    codec_ctx_->height = out_height;
    // codec_ctx_->time_base = (AVRational){1, static_cast<int>(cap_fps_)};
    // codec_ctx_->framerate = (AVRational){static_cast<int>(cap_fps_), 1};
    codec_ctx_->time_base = (AVRational){1, video_fps_};
    codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    // codec_ctx_->profile = 4; // High

    /* open it */
    ret = avcodec_open2(codec_ctx_, codec, NULL);
    if (ret < 0)
    {
      SPDLOG_CRITICAL("Could not open codec");
      return -1;
    }

    frame_ = av_frame_alloc();
    if (!frame_)
    {
      SPDLOG_CRITICAL("Could not allocate video frame");
      return -1;
    }
    frame_->format = codec_ctx_->pix_fmt;
    frame_->width = codec_ctx_->width;
    frame_->height = codec_ctx_->height;

    ret = av_frame_get_buffer(frame_, 0);
    if (ret < 0)
    {
      SPDLOG_CRITICAL("Could not allocate the video frame data");
      return -1;
    }

    // https://gist.github.com/foowaa/1d296a9dee81c7a2a52f291c95e55680
    conversion_ = sws_getContext(
        out_width, out_height, AV_PIX_FMT_BGR24,
        out_width, out_height, AV_PIX_FMT_YUV420P,
        SWS_FAST_BILINEAR, NULL, NULL, NULL);
    SPDLOG_INFO("Inited");
    return 0;
  }
  void Camera::DelEncoder()
  {
    /* flush the encoder */
    if (codec_ctx_ && packet)
    {
      avcodec_send_frame(codec_ctx_, NULL);
      GetEncoded();
    }
    if (codec_ctx_)
    {
      avcodec_free_context(&codec_ctx_);
    }
    if (frame_)
    {
      av_frame_free(&frame_);
    }
    if (packet)
    {
      av_packet_free(&packet);
    }
    if (conversion_)
    {
      sws_freeContext(conversion_);
    }
  }
  int Camera::GetEncoded()
  {
    int ret = 0;

    while (ret >= 0)
    {
      ret = avcodec_receive_packet(codec_ctx_, packet);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
        return 0;
      else if (ret < 0)
      {
        SPDLOG_ERROR("Error during encoding");
        return -1;
      }

      SPDLOG_TRACE("Read packet {} (size={})", packet->pts, packet->size);
      // fwrite(packet->data, 1, packet->size, f);
      av_packet_unref(packet);
    }

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
    cv::resize(mat_cap_, mat_resize_, cv::Size(out_width, out_height), 0, 0, cv::INTER_NEAREST);
    SPDLOG_TRACE("Encode image");
    // encoded.clear();
    // cv::imencode(".jpg", mat_resize_, encoded);
    const int cvLinesizes[1] = {static_cast<int>(mat_resize_.step1())};
    // TODO: memcopy?
    sws_scale(conversion_, &mat_resize_.data, cvLinesizes, 0, out_height,
              frame_->data, frame_->linesize);
    frame_->pts = video_time_;
    video_time_++;
    SPDLOG_TRACE("Send frame {} ", frame_->pts);
    ret = avcodec_send_frame(codec_ctx_, frame_);
    if (ret < 0)
    {
      SPDLOG_ERROR("Error sending a frame for encoding");
      return -1;
    }
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