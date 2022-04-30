#ifndef AUTODRONE_RPI4_CAMERA
#define AUTODRONE_RPI4_CAMERA

#include <string>
#include <vector>

#include <opencv2/videoio.hpp>

namespace rpi4
{
  class Config;

  class Camera
  {
  private:
    Config *config_ = nullptr;
    int cap_device_;
    double cap_fps_;
    int out_width_;
    int out_height_;
    int jpg_quality_;
    cv::VideoCapture cap_;
    std::vector<int> jpg_params_;
    cv::Mat mat_cap_;
    cv::Mat mat_resize_;

  public:
    int cap_width;
    int cap_height;

    std::vector<uchar> encoded;

  public:
    Camera();
    void LoadConfig(Config *config);
    int Open();
    int Capture(cv::Mat &mat);
    void SetOutputSize(int width, int height);
    bool IsOpened();
  };

} // namespace rpi4
#endif // AUTODRONE_RPI4_CAMERA