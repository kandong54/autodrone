#ifndef AUTODRONE_RPI4_TFLITE
#define AUTODRONE_RPI4_TFLITE

#include <memory>
#include <vector>

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
    float input_quant_scale_ = 1;
    int32_t input_quant_zero_point_ = 0;
    TfLiteType input_type_ = kTfLiteNoType;
    float output_quant_scale_ = 1;
    int32_t output_quant_zero_point_ = 0;
    TfLiteType output_type_ = kTfLiteNoType;
    int input_height_ = 0;
    int input_width_ = 0;
    int input_channels_ = 0;
    int output_nums_ = 0;
    size_t input_bytes_ = 0;
    size_t output_bytes_ = 0;
    std::vector<float> prediction_;
    float threshold_ = 0.5;

  public:
    TFLite(/* args */);
    ~TFLite();
    bool Load();
    bool Inference(cv::Mat &image);
  };

} // namespace rpi4
#endif // AUTODRONE_RPI4_TFLITE