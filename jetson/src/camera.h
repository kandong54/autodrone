#ifndef AUTODRONE_JETSON_CAMERA
#define AUTODRONE_JETSON_CAMERA

#include <NvJpegDecoder.h>
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
  unsigned int size;
  /* File descriptor of NvBuffer */
  int dmabuff_fd;
} nv_buffer;

class Camera {
 private:
  YAML::Node &config_;
  Model &model_;
  int cam_fd_ = -1;
  int model_size_;
  unsigned int sizeimage_;
  NvJPEGDecoder *jpegdec_;
  EGLDisplay egl_display_;
  float3 *rgb_image_;
  float3 *resize_image_;
  float3 *model_image_;
  NvBufferTransformParams transParams_;
  nv_buffer trans_buffer_;
  nv_buffer in_jpeg_buffer_;
  
 public:
  static const int mjpeg_num = 2;
  nv_buffer out_jpeg_buffer[mjpeg_num];
  unsigned int mjpeg_index = 0;

 public:
  Camera(YAML::Node &config, Model &model);
  int Open();
  int Capture();
  int Convert(int fd);
  ~Camera();
};

}  // namespace jetson
#endif  // AUTODRONE_JETSON_CAMERA