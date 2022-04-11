#ifndef AUTODRONE_RPI4_DRONE_APP
#define AUTODRONE_RPI4_DRONE_APP

#include <memory>
#include <thread>

namespace rpi4
{
  class Camera;
  class TFLite;

  class DroneApp
  {
  private:
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<TFLite> tflite_;
    std::unique_ptr<std::thread> thread_;

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