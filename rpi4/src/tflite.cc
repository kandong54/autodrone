#include "tflite.h"

#include <thread>

#include <opencv2/dnn/dnn.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
// #include "tensorflow/lite/optional_debug_tools.h"
#include <spdlog/spdlog.h>

#include "config.h"
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc
// https://www.tensorflow.org/lite/api_docs/cc/class/tflite/interpreter
// https://gist.github.com/WesleyCh3n/ce652db395668ec64fe0ca6fa0e55d0d

namespace rpi4
{

  TFLite::TFLite(/* args */)
  {
    // TODO: remove default value
    model_path_ = "./model/best-int8.tflite";
    conf_threshold_ = 0.25;
    iou_threshold_ = 0.45;
  }

  TFLite::~TFLite()
  {
  }
  void TFLite::LoadConfig(Config *config)
  {
    if (config->node["detector"])
    {
      if (config->node["detector"]["model_path"])
      {
        model_path_ = config->node["detector"]["model_path"].as<std::string>();
      }
      if (config->node["detector"]["confidence_threshold"])
      {
        conf_threshold_ = config->node["detector"]["confidence_threshold"].as<float>();
      }
      if (config->node["detector"]["iou_threshold"])
      {
        iou_threshold_ = config->node["detector"]["iou_threshold"].as<float>();
      }
    }
  }
  void TFLite::SetCameraSize(int width, int height)
  {
    camera_width_ = width;
    camera_height_ = height;
  }
  int TFLite::Load()
  {
    // Create model from file. Note that the model instance must outlive the
    // interpreter instance.
    SPDLOG_INFO("Loading model {}", model_path_);
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str());
    if (model_ == nullptr)
    {
      SPDLOG_CRITICAL("Failed to mmap  model {}!", model_path_);
      return -1;
    }
    // Create an Interpreter with an InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    if (tflite::InterpreterBuilder(*model_, resolver)(&interpreter_) != kTfLiteOk)
    {
      SPDLOG_CRITICAL("Failed to build interpreter!");
      return -1;
    }
    if (interpreter_->AllocateTensors() != kTfLiteOk)
    {
      SPDLOG_CRITICAL("Failed to allocate tensors!");
      return -1;
    }
    // get input tensor metadata
    TfLiteTensor *input_tensor = interpreter_->input_tensor(0);
    TfLiteTensor *output_tensor = interpreter_->output_tensor(0);
    input_data_ptr_ = &input_tensor->data;
    // quantization
    if (input_tensor->quantization.type == kTfLiteAffineQuantization)
    {
      is_quantization_ = true;
      // TODO: use TfLiteAffineQuantization
      // auto params = static_cast<TfLiteAffineQuantization *>(input_tensor->quantization.params);
      auto input_params = input_tensor->params;
      input_quant_scale_ = input_params.scale;
      input_quant_zero_point_ = input_params.zero_point;
      auto output_params = output_tensor->params;
      output_quant_scale_ = output_params.scale;
      output_quant_zero_point_ = output_params.zero_point;
    }
    // Type
    input_type_ = input_tensor->type;
    output_type_ = output_tensor->type;
    // dims
    // TODO: send to camera
    TfLiteIntArray *input_dims = input_tensor->dims;
    input_height = input_dims->data[1];
    input_width = input_dims->data[2];
    input_channels_ = input_dims->data[3];
    // reserve prediction space
    // [1, 25200, 6]
    // [Cx, Cy, width , height, confidence, class]
    TfLiteIntArray *output_dims = output_tensor->dims;
    output_nums_ = output_dims->data[1];
    class_nums_ = output_dims->data[2] - kClass;
    boxes.reserve(output_dims->data[1]);
    confs.reserve(output_dims->data[1]);
    class_id.reserve(output_dims->data[1]);
    // SetNumThreads
    unsigned int threads_num = std::thread::hardware_concurrency();
    interpreter_->SetNumThreads(threads_num);
    SPDLOG_INFO("Set {} Threads", threads_num);
    // Warmup
    SPDLOG_INFO("Warm up");
    if (interpreter_->Invoke() != kTfLiteOk)
    {
      SPDLOG_CRITICAL("Failed to invoke interpreter!");
      return -1;
    }
    SPDLOG_INFO("Loaded");
    return 0;
  }

  bool TFLite::IsWork()
  {
    return (interpreter_->Invoke() == kTfLiteOk);
  }

  template <class T>
  void TFLite::AddPrediction()
  {
    // https://github.com/itsnine/yolov5-onnxruntime/blob/master/src/detector.cpp
    boxes.clear();
    confs.clear();
    class_id.clear();
    indices.clear();
    T quant_threshold = conf_threshold_ / output_quant_scale_ + output_quant_zero_point_;
    T *output_tensor_ptr = interpreter_->typed_output_tensor<T>(0);
    int step = static_cast<int>(kClass) + class_nums_;
    for (int i = 0; i < output_nums_ * step; i += step)
    {
      // [Cx, Cy, width , height, confidence, class1, class2, ...]
      if (output_tensor_ptr[i + kConfidence] >= quant_threshold)
      {
        int center_x = static_cast<int>((static_cast<float>(output_tensor_ptr[i + kXCenter]) - output_quant_zero_point_) * output_quant_scale_ * camera_width_);
        int center_y = static_cast<int>((static_cast<float>(output_tensor_ptr[i + kYCenter]) - output_quant_zero_point_) * output_quant_scale_ * camera_height_);
        int width = static_cast<int>((static_cast<float>(output_tensor_ptr[i + kWidth]) - output_quant_zero_point_) * output_quant_scale_ * camera_width_);
        int height = static_cast<int>((static_cast<float>(output_tensor_ptr[i + kHeight]) - output_quant_zero_point_) * output_quant_scale_ * camera_height_);
        int left = center_x - width / 2;
        int top = center_y - height / 2;
        int id = std::distance(output_tensor_ptr + i + kClass, std::max_element(output_tensor_ptr + i + kClass, output_tensor_ptr + i + kClass + class_nums_));
        float confidence = (static_cast<float>(output_tensor_ptr[i + kConfidence]) - output_quant_zero_point_) * output_quant_scale_;
        float confidence_class = (static_cast<float>(output_tensor_ptr[i + kClass + id]) - output_quant_zero_point_) * output_quant_scale_;
        boxes.emplace_back(left, top, width, height);
        confs.emplace_back(confidence * confidence_class);
        class_id.emplace_back(id);
      }
    }
    // NMS
    cv::dnn::NMSBoxes(boxes, confs, conf_threshold_, iou_threshold_, indices);
  }

  int TFLite::Inference(cv::Mat frame)
  {
    // TODO: Check interpreter
    // Type convert, default CV_8UC3, kTfLiteUInt8
    SPDLOG_TRACE("Convert");
    switch (input_type_)
    {
    case kTfLiteUInt8:
    {
      if (is_quantization_)
      {
        frame.convertTo(mat_convert, CV_8UC3, 1 / (input_quant_scale_ * 255), input_quant_zero_point_);
      }
      else
      {
        frame.convertTo(mat_convert, CV_8UC3, 1 / 255.0F);
      }
      break;
    }
    case kTfLiteFloat32:
    {
      frame.convertTo(mat_convert, CV_32FC3, 1 / 255.0F);
      break;
    }
    case kTfLiteFloat16:
    {
      frame.convertTo(mat_convert, CV_16FC3, 1 / 255.0F);
      break;
    }
    default:
    {
      SPDLOG_CRITICAL("Unsupported type: ", input_type_);
      return -1;
      break;
    }
    }
    // Load data
    input_data_ptr_->data = mat_convert.data;
    // Run!
    SPDLOG_TRACE("Invoke");
    interpreter_->Invoke();
    SPDLOG_TRACE("Output");
    switch (output_type_)
    {
    case kTfLiteUInt8:
    {
      AddPrediction<unsigned char>();
      break;
    }
    case kTfLiteFloat32:
    {
      AddPrediction<float>();
      break;
    }
    // case kTfLiteFloat16:
    // {
    //   AddPrediction<TfLiteFloat16>();
    //   break;
    // }
    default:
    {
      SPDLOG_CRITICAL("Unsupported type: ", input_type_);
      return -1;
      break;
    }
    }
    SPDLOG_TRACE("Total: {}", indices.size());

    return 0;
  }

} // namespace rpi4