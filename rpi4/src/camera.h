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

  public:
    int out_width;
    int out_height;
    std::vector<uchar> encoded;

  public:
    Camera();
    int Open();
    int Capture(cv::Mat &mat);
    bool IsOpened();
  };

} // namespace rpi4
#endif // AUTODRONE_RPI4_CAMERA