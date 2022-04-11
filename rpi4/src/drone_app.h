#ifndef AUTODRONE_RPI4_DRONE_APP
#define AUTODRONE_RPI4_DRONE_APP

#include <memory>
#include <thread>

#include <opencv2/core.hpp>

namespace rpi4
{
  class Camera;
  class TFLite;

  class DroneApp
  {
  private: 
    std::unique_ptr<std::thread> thread_;

  public:
    std::unique_ptr<Camera> camera;
    std::unique_ptr<TFLite> tflite;
    cv::Mat frame;

  private:
    void Run();

  public:
    DroneApp(/* args */);
    ~DroneApp();
    bool IsConnected();
    void BuildAndStart();
  };

} // namespace rpi4
#endif // AUTODRONE_RPI4_DRONE_APP