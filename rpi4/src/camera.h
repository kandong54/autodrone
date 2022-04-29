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
    double cap_fps_;
    cv::Mat mat_cap_;
    cv::Mat mat_resize_;
    int out_width_;
    int out_height_;
    int jpg_quality_;
    std::vector<int> jpg_params_;

  public:
    int cap_width;
    int cap_height;

    std::vector<uchar> encoded;

  public:
    Camera();
    int Open();
    int Capture(cv::Mat &mat);
    bool IsOpened();
  };

} // namespace rpi4
#endif // AUTODRONE_RPI4_CAMERA