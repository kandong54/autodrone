#ifndef AUTODRONE_JETSON_CAMERA
#define AUTODRONE_JETSON_CAMERA

#include <NvJpegDecoder.h>
#include <NvJpegEncoder.h>
#include <cuda_runtime.h>
#include <nvbuf_utils.h>
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
  NvBufferTransformParams transParams_; // used by Detect & Encode
  nv_buffer detect_rbg_buffer_;
  // Encode
  NvJPEGEncoder *jpegenc_;
  nv_buffer encode_yuv_buffer_;
  int encode_jpeg_max_size_;
  int encode_quality_;
  // Depth
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
  ~Camera();
};

}  // namespace jetson
#endif  // AUTODRONE_JETSON_CAMERA