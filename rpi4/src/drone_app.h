#ifndef AUTODRONE_RPI4_DRONE_APP
#define AUTODRONE_RPI4_DRONE_APP

#include <memory>
#include <thread>
#include <condition_variable>

#include "camera.h"
#include "tflite.h"

namespace rpi4
{
  class Config;

  class DroneApp
  {
  private:
    std::thread thread_;

  public:
    Camera camera;
    TFLite tflite;
    std::condition_variable cv;
    bool cv_flag;

  private:
    void Run();

  public:
    DroneApp(/* args */);
    ~DroneApp();
    bool IsConnected();
    int BuildAndStart(Config *config);
  };

} // namespace rpi4
#endif // AUTODRONE_RPI4_DRONE_APP