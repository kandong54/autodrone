#ifndef AUTODRONE_RPI4_CAMERA
#define AUTODRONE_RPI4_CAMERA

#include <thread>
#include <mutex>

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

    std::unique_lock<std::mutex> lock_;

  public:
    std::mutex mutex;
    int out_width;
    int out_height;

  public:
    Camera();
    bool Open();
    bool CaptureImage(cv::Mat &frame);
    bool IsOpened();
  };

} // namespace rpi4
#endif // AUTODRONE_RPI4_CAMERA