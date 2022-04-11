#include "tflite.h"

#include <thread>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
// #include "tensorflow/lite/optional_debug_tools.h"

#include "log.h"

// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc
// https://www.tensorflow.org/lite/api_docs/cc/class/tflite/interpreter
// https://gist.github.com/WesleyCh3n/ce652db395668ec64fe0ca6fa0e55d0d

namespace rpi4
{
  TFLite::TFLite(/* args */)
  {
    // Init parameter
    model_path_ = "./bin/best-int8.tflite";
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
    auto input_tensor = interpreter_->input_tensor(0);
    // quantization
    if (input_tensor->quantization.type == kTfLiteAffineQuantization)
    {
      is_quantization_ = true;
      // TODO: use TfLiteAffineQuantization
      // auto params = static_cast<TfLiteAffineQuantization *>(input_tensor->quantization.params);
      auto params = input_tensor->params;
      quant_scale_ = params.scale;
      quant_zero_point_ = params.zero_point;
    }
    // Type
    input_type_ = input_tensor->type;
    //Bytes
    input_bytes_ = input_tensor->bytes;
    // dims
    // TODO: send to camera
    TfLiteIntArray *dims = input_tensor->dims;
    input_height_ = dims->data[1];
    input_width_ = dims->data[2];
    input_channels_ = dims->data[3];
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

  bool TFLite::Inference(cv::Mat &image)
  {
    SPDLOG_TRACE("Read");
    // TODO: Check interpreter
    cv::Mat tmp_img;
    image.copyTo(tmp_img);
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
        tmp_img.convertTo(tmp_img, CV_32FC3, 1 / 255.0);
        tmp_img.convertTo(tmp_img, CV_8UC3, 1 / quant_scale_, quant_zero_point_);
      }
      input_tensor_ptr = interpreter_->typed_input_tensor<unsigned char>(0);
      tmp_img_ptr = tmp_img.ptr<unsigned char>(0);
      break;
    }
    case kTfLiteFloat32:
    {
      tmp_img.convertTo(tmp_img, CV_32FC3, 1 / 255.0);
      input_tensor_ptr = interpreter_->typed_input_tensor<float>(0);
      tmp_img_ptr = tmp_img.ptr<float>(0);
      break;
    }
    case kTfLiteFloat16:
    {
      tmp_img.convertTo(tmp_img, CV_16FC3, 1 / 255.0);
      input_tensor_ptr = interpreter_->typed_input_tensor<TfLiteFloat16>(0);
      tmp_img_ptr = tmp_img.ptr<uint16_t>(0);
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
    SPDLOG_TRACE("Finish");
    return true;
  }
} // namespace rpi4