#ifndef AUTODRONE_RPI4_CAMERA
#define AUTODRONE_RPI4_CAMERA

#include <opencv2/videoio.hpp>

namespace rpi4
{

  class Camera
  {
  private:
    cv::VideoCapture cap_;

  public:
    Camera();
    bool CaptureImage();
  };

} // namespace rpi4
#endif // AUTODRONE_RPI4_CAMERA