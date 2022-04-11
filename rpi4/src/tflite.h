#ifndef AUTODRONE_RPI4_TFLITE
#define AUTODRONE_RPI4_TFLITE

#include <memory>

#include <opencv2/core.hpp>
#include <tensorflow/lite/interpreter.h>
#include "tensorflow/lite/model.h"

namespace rpi4
{
  class TFLite
  {
  private:
    std::string model_path_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    std::unique_ptr<tflite::FlatBufferModel> model_;
    bool is_quantization_ = false;
    float quant_scale_;
    int32_t quant_zero_point_;
    TfLiteType input_type_;
    int input_height_;
    int input_width_;
    int input_channels_;
    size_t input_bytes_;

  public:
    TFLite(/* args */);
    ~TFLite();
    bool Load();
    bool Inference(cv::Mat &image);
  };

} // namespace rpi4
#endif // AUTODRONE_RPI4_TFLITE