#ifndef AUTODRONE_JETSON_CAMERA
#define AUTODRONE_JETSON_CAMERA

#include <NvJpegDecoder.h>
#include <NvJpegEncoder.h>
#include <cuda_runtime.h>
#include <nvbuf_utils.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <yaml-cpp/yaml.h>

namespace jetson {

class Model;

// /usr/src/jetson_multimedia_api/samples/12_camera_v4l2_cuda/camera_v4l2_cuda.h
typedef struct
{
  /* User accessible pointer */
  unsigned char *start;
  /* Buffer length */
  unsigned long size;
  /* File descriptor of NvBuffer */
  int dmabuff_fd;
} nv_buffer;

class Camera {
 private:
  YAML::Node &config_;
  Model &model_;
  // Capture
  int cam_fd_ = -1;
  NvJPEGDecoder *jpegdec_;
  int capture_yuv_fd_;
  nv_buffer capture_jpeg_buffer_;
  // Detect
  int detect_rbg_fd_;
  // Encode
  NvJPEGEncoder *jpegenc_;
  int encode_yuv_fd_;
  int encode_jpeg_max_size_;
  int encode_quality_;
  // Depth
  int depth_factor_;
  int camera_width_;
  NvBufferTransformParams depth_left_trans_;
  NvBufferTransformParams depth_right_trans_;
  int depth_left_fd_;
  int depth_right_fd_;
  VPIImage depth_left_img_ = NULL;
  VPIImage depth_right_img_ = NULL;
  VPIImage depth_left_Y16_img_ = NULL;
  VPIImage depth_right_Y16_img_ = NULL;
  VPIStream depth_stream_ = NULL;
  VPIPayload depth_stereo_ = NULL;
  void * depth_disparity_data_;
  void * depth_confidenceMap_data_; 
  VPIImage depth_disparity_ = NULL;
  VPIImage depth_confidenceMap_ = NULL;
  VPIConvertImageFormatParams depth_convParams_;

 public:
  static const int mjpeg_num = 2;
  nv_buffer encode_jpeg_list[mjpeg_num];
  unsigned int encode_index = 0;


 public:
  Camera(YAML::Node &config, Model &model);
  int Open();
  int Capture();
  int Detect();
  int Encode();
  int Depth();
  int PostDepth();  
  int RunParallel();
  ~Camera();
};

}  // namespace jetson
#endif  // AUTODRONE_JETSON_CAMERA