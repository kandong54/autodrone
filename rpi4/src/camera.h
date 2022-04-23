#ifndef AUTODRONE_RPI4_CAMERA
#define AUTODRONE_RPI4_CAMERA

#include <string>

#include <opencv2/videoio.hpp>

namespace rpi4
{

  class Camera
  {
  private:
    cv::VideoCapture cap_;

    int cap_device_;
    int cap_width_;
    int cap_height_;
    int cap_fps_;

  public:
    int out_width;
    int out_height;

  public:
    Camera();
    int Open();
    int Capture(cv::Mat &frame);
    bool IsOpened();
    void Compress(cv::Mat img, std::vector<uchar> &buf);
  };

} // namespace rpi4
#endif // AUTODRONE_RPI4_CAMERA