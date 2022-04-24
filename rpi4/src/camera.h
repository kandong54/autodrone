#ifndef AUTODRONE_RPI4_CAMERA
#define AUTODRONE_RPI4_CAMERA

#include <string>
#include <vector>

#include <opencv2/videoio.hpp>
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

namespace rpi4
{

  class Camera
  {
  private:
    cv::VideoCapture cap_;

    int cap_device_;
    int cap_width_;
    int cap_height_;
    double cap_fps_;
    cv::Mat mat_cap_;
    cv::Mat mat_resize_;
    int bit_rate_;
    // hardware defauft FPS
    const int video_fps_ = 30;

    AVCodecContext *codec_ctx_ = nullptr;
    int codec_fd_;
    AVFrame *frame_ = nullptr;
    SwsContext *conversion_ = nullptr;

  public:
    int out_width;
    int out_height;
    // std::vector<uchar> encoded;
    AVPacket *packet = nullptr;
    int h264_i_frame_period = 10;
    int64_t video_timestamp;

  public:
    Camera();
    ~Camera();
    int Open();
    int Capture(cv::Mat &mat);
    bool IsOpened();
    int GetEncoded();
    int InitEncoder();
    void DelEncoder();
  };

} // namespace rpi4
#endif // AUTODRONE_RPI4_CAMERA