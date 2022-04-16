#include "tflite.h"

#include <thread>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include <opencv2/imgproc.hpp>
// #include "tensorflow/lite/optional_debug_tools.h"
#include <spdlog/spdlog.h>

// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc
// https://www.tensorflow.org/lite/api_docs/cc/class/tflite/interpreter
// https://gist.github.com/WesleyCh3n/ce652db395668ec64fe0ca6fa0e55d0d

namespace rpi4
{

  TFLite::TFLite(/* args */)
  {
    // Init parameter
    model_path_ = "./model/best-int8.tflite";
    threshold_ = 0.5;
    lock_ = std::unique_lock<std::mutex>(mutex, std::defer_lock);
  }

  TFLite::~TFLite()
  {
  }

  bool TFLite::Load()
  {
    // Create model from file. Note that the model instance must outlive the
    // interpreter instance.
    SPDLOG_INFO("Loading model {}", model_path_);
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str());
    if (model_ == nullptr)
    {
      SPDLOG_CRITICAL("Failed to mmap  model {}!", model_path_);
      return false;
    }
    // Create an Interpreter with an InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    if (tflite::InterpreterBuilder(*model_, resolver)(&interpreter_) != kTfLiteOk)
    {
      SPDLOG_CRITICAL("Failed to build interpreter!");
      return false;
    }
    if (interpreter_->AllocateTensors() != kTfLiteOk)
    {
      SPDLOG_CRITICAL("Failed to allocate tensors!");
      return false;
    }
    // get input tensor metadata
    TfLiteTensor *input_tensor = interpreter_->input_tensor(0);
    TfLiteTensor *output_tensor = interpreter_->output_tensor(0);
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
    // Bytes
    input_bytes_ = input_tensor->bytes;
    output_bytes_ = output_tensor->bytes;
    // dims
    // TODO: send to camera
    TfLiteIntArray *input_dims = input_tensor->dims;
    input_height_ = input_dims->data[1];
    input_width_ = input_dims->data[2];
    input_channels_ = input_dims->data[3];
    // reserve prediction space
    // [1, 25200, 6]
    // [Cx, Cy, width , height, confidence, class]
    TfLiteIntArray *output_dims = output_tensor->dims;
    output_nums_ = output_dims->data[1];
    prediction.reserve(output_dims->data[1] * output_dims->data[2]);
    // SetNumThreads
    unsigned int threads_num = std::thread::hardware_concurrency();
    interpreter_->SetNumThreads(threads_num);
    SPDLOG_INFO("Set {} Threads", threads_num);
    // Warmup
    SPDLOG_INFO("Warm up");
    if (interpreter_->Invoke() != kTfLiteOk)
    {
      SPDLOG_CRITICAL("Failed to invoke interpreter!");
      return false;
    }
    SPDLOG_INFO("Loaded model {}", model_path_);
    return true;
  }
  bool TFLite::IsWork()
  {
    return (interpreter_->Invoke() == kTfLiteOk);
  }
  namespace
  {
    template <class T>
    void AddPrediction(std::unique_ptr<tflite::Interpreter> const &interpreter, std::vector<float> &prediction, float threshold, int output_nums)
    {
      T *output_tensor_ptr = interpreter->typed_output_tensor<T>(0);
      for (int i = 0; i < output_nums; i++)
      {
        if (output_tensor_ptr[i * kOutputNum + kConfidence] >= threshold)
        {
          for (size_t j = 0; j < kOutputNum; j++)
          {
            prediction.push_back(static_cast<float>(output_tensor_ptr[i * kOutputNum + j]));
          }
        }
      }
    }

  } // namespace

  bool TFLite::Inference(cv::Mat &frame)
  {
    SPDLOG_TRACE("Read");
    // TODO: Check interpreter
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    // Type convert, default CV_8UC3, kTfLiteUInt8
    SPDLOG_TRACE("Convert");
    void *input_tensor_ptr = nullptr;
    void *tmp_img_ptr = nullptr;
    switch (input_type_)
    {
    case kTfLiteUInt8:
    {
      if (is_quantization_)
      {
        frame.convertTo(frame, CV_32FC3, 1 / 255.0);
        frame.convertTo(frame, CV_8UC3, 1 / input_quant_scale_, input_quant_zero_point_);
      }
      input_tensor_ptr = interpreter_->typed_input_tensor<unsigned char>(0);
      tmp_img_ptr = frame.ptr<unsigned char>(0);
      break;
    }
    case kTfLiteFloat32:
    {
      frame.convertTo(frame, CV_32FC3, 1 / 255.0);
      input_tensor_ptr = interpreter_->typed_input_tensor<float>(0);
      tmp_img_ptr = frame.ptr<float>(0);
      break;
    }
    case kTfLiteFloat16:
    {
      frame.convertTo(frame, CV_16FC3, 1 / 255.0);
      input_tensor_ptr = interpreter_->typed_input_tensor<TfLiteFloat16>(0);
      tmp_img_ptr = frame.ptr<uint16_t>(0);
      break;
    }
    default:
    {
      SPDLOG_CRITICAL("Unsupported type: ", input_type_);
      return false;
      break;
    }
    }
    // Load data
    SPDLOG_TRACE("Memcpy");
    memcpy(input_tensor_ptr, tmp_img_ptr, input_bytes_);
    // Run!
    SPDLOG_TRACE("Invoke");
    interpreter_->Invoke();
    SPDLOG_TRACE("Output");
    lock_.lock();
    prediction.clear();
    // TODO: width and height
    // [1, 25200, 6]
    // [Cx, Cy, width , height, confidence, class]
    if (is_quantization_)
    {
      unsigned char quant_threshold = threshold_ / output_quant_scale_ + output_quant_zero_point_;
      unsigned char *output_tensor_ptr = interpreter_->typed_output_tensor<unsigned char>(0);
      for (int i = 0; i < output_nums_; i++)
      {
        if (output_tensor_ptr[i * kOutputNum + kConfidence] >= quant_threshold)
        {
          for (size_t j = 0; j < kOutputNum; j++)
          {
            prediction.push_back((static_cast<float>(output_tensor_ptr[i * kOutputNum + j]) - output_quant_zero_point_) * output_quant_scale_);
          }
        }
      }
    }
    else
    {
      switch (output_type_)
      {
      case kTfLiteUInt8:
      {
        AddPrediction<unsigned char>(interpreter_, prediction, threshold_, output_nums_);
        break;
      }
      case kTfLiteFloat32:
      {
        AddPrediction<float>(interpreter_, prediction, threshold_, output_nums_);
        break;
      }
      // case kTfLiteFloat16:
      // {
      //   AddPrediction<TfLiteFloat16>(interpreter_, prediction, threshold_, output_nums_);
      //   break;
      // }
      default:
      {
        SPDLOG_CRITICAL("Unsupported type: ", input_type_);
        return false;
        break;
      }
      }
    }
    lock_.unlock();
    SPDLOG_DEBUG("Total: {}", prediction.size() / kOutputNum);

    return true;
  }

} // namespace rpi4