#ifndef AUTODRONE_RPI4_DRONE_APP
#define AUTODRONE_RPI4_DRONE_APP

#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <opencv2/core.hpp>

#include "camera.h"
#include "tflite.h"

namespace rpi4
{
  class DroneApp
  {
  private:
    std::unique_ptr<std::thread> thread_;

  public:
    std::unique_ptr<Camera> camera;
    std::unique_ptr<TFLite> tflite;
    std::vector<uchar> frame;
    std::condition_variable cv;
    bool cv_flag;

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