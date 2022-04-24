#ifndef AUTODRONE_RPI4_TFLITE
#define AUTODRONE_RPI4_TFLITE

#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

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
    int class_nums_ = 0;
    size_t input_bytes_ = 0;
    size_t output_bytes_ = 0;
    float conf_threshold_ = 0.25;
    float iou_threshold_ = 0.45;
    TfLitePtrUnion *input_data_ptr_ = nullptr;
    int cap_width_;
    int cap_height_;
    cv::Mat mat_convert;

  public:
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> class_id;
    std::vector<int> indices;

  private:
    template <class T>
    void AddPrediction();

  public:
    TFLite(/* args */);
    ~TFLite();
    int Load();
    int Inference(cv::Mat image);
    bool IsWork();
  };

  // [Cx, Cy, width , height, confidence, class]
  enum OutputArray
  {
    kXCenter = 0,
    kYCenter = 1,
    kWidth = 2,
    kHeight = 3,
    kConfidence = 4,
    kClass = 5
  };
} // namespace rpi4
#endif // AUTODRONE_RPI4_TFLITE