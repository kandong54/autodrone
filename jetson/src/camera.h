#ifndef AUTODRONE_JETSON_CAMERA
#define AUTODRONE_JETSON_CAMERA

#include <NvJpegDecoder.h>
#include <NvJpegEncoder.h>
#include <yaml-cpp/yaml.h>
#include <nvbuf_utils.h>

namespace jetson {

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
  // Capture
  int cam_fd_ = -1;
  NvJPEGDecoder *jpegdec_;
  nv_buffer capture_buffer_;
  // Encode
  NvJPEGEncoder *jpegenc_;
  int encode_jpeg_max_size_;
  int encode_quality_;
  // detector & depth
  NvBufferTransformParams transform_params_;

 public:
  int detector_fd;
  int depth_fd;
  // 2 buffers for server to read
  // avoid potential data race
  // the main thread and server thread will always write/read different buffer
  static const int mjpeg_num = 2;
  nv_buffer encode_jpeg_list[mjpeg_num];
  unsigned int encode_index = 0;

 public:
  Camera(YAML::Node &config);
  int Open();
  int Capture();
  int Encode();
  ~Camera();
};

}  // namespace jetson
#endif  // AUTODRONE_JETSON_CAMERA