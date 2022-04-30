#ifndef AUTODRONE_RPI4_TFLITE
#define AUTODRONE_RPI4_TFLITE

#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

namespace rpi4
{
  class Config;

  // TODO: decouple it to class Detector, interface Model, class TFlite
  class TFLite
  {
  private:
    Config *config_;
    std::string model_path_;
    float conf_threshold_ = 0;
    float iou_threshold_ = 0;

    std::unique_ptr<tflite::Interpreter> interpreter_;
    std::unique_ptr<tflite::FlatBufferModel> model_;
    bool is_quantization_ = false;
    float input_quant_scale_ = 1;
    int32_t input_quant_zero_point_ = 0;
    TfLiteType input_type_ = kTfLiteNoType;
    float output_quant_scale_ = 1;
    int32_t output_quant_zero_point_ = 0;
    TfLiteType output_type_ = kTfLiteNoType;
    int input_channels_ = 0;
    int output_nums_ = 0;
    int class_nums_ = 0;
    TfLitePtrUnion *input_data_ptr_ = nullptr;
    int camera_width_ = 0;
    int camera_height_ = 0;
    cv::Mat mat_convert;

  public:
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> class_id;
    std::vector<int> indices;
    int input_height = 0;
    int input_width = 0;

  private:
    template <class T>
    void AddPrediction();

  public:
    TFLite(/* args */);
    ~TFLite();
    void LoadConfig(Config *config);
    int Load();
    int Inference(cv::Mat image);
    bool IsWork();
    void SetCameraSize(int width, int height);
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