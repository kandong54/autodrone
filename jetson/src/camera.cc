#include "camera.h"

// #include "model.h"

// Ref: /usr/src/jetson_multimedia_api/samples/
//      https://github.com/dusty-nv/jetson-utils/blob/master/codec/gstBufferManager.cpp
//      https://github.com/NVIDIA-AI-IOT/jetson-stereo-depth/blob/master/detph_pipeline_cpp/main.cpp

#include <asm/types.h> /* for videodev2.h */
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace jetson {

Camera::Camera(YAML::Node& config) : config_(config) {
  memset(encode_jpeg_list, 0, mjpeg_num * sizeof(nv_buffer));
  const int camera_width = config_["camera"]["width"].as<unsigned int>();
  const int camera_height = config_["camera"]["height"].as<unsigned int>();
  // how to set the maximal size of jpeg
  encode_jpeg_max_size_ = camera_width * camera_height * 3;
}

int Camera::Open() {
  // const
  const int model_size = config_["detector"]["size"].as<unsigned int>();
  const int depth_size = config_["depth"]["size"].as<unsigned int>();
  const int camera_width = config_["camera"]["width"].as<unsigned int>();
  const int camera_height = config_["camera"]["height"].as<unsigned int>();

  // open_device
  // camera
  std::string dev_name = config_["camera"]["device"].as<std::string>();
  cam_fd_ = open(dev_name.c_str(), O_RDWR /* required */ | O_NONBLOCK, 0);
  if (-1 == cam_fd_)
    SPDLOG_CRITICAL("Failed to open {}", dev_name);
  // NvJPEGDecoder
  jpegdec_ = NvJPEGDecoder::createJPEGDecoder("jpegdec");
  if (!jpegdec_)
    SPDLOG_CRITICAL("Failed to create NvJPEGDecoder");
  // NvJPEGEncoder
  jpegenc_ = NvJPEGEncoder::createJPEGEncoder("jpegenc");
  if (!jpegenc_)
    SPDLOG_CRITICAL("Failed to create NvJPEGEncoder");

  // init_device
  struct v4l2_capability cap;
  // output supported mode
  ioctl(cam_fd_, VIDIOC_QUERYCAP, &cap);
  if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)
    SPDLOG_INFO("V4L2_CAP_VIDEO_CAPTURE");
  if (cap.capabilities & V4L2_CAP_READWRITE)
    SPDLOG_INFO("V4L2_CAP_READWRITE");
  if (cap.capabilities & V4L2_CAP_ASYNCIO)
    SPDLOG_INFO("V4L2_CAP_ASYNCIO");
  if (cap.capabilities & V4L2_CAP_STREAMING)
    SPDLOG_INFO("V4L2_CAP_STREAMING");

  // image size
  struct v4l2_format fmt;
  memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  // set format
  // v4l2-ctl --list-formats-ext
  fmt.fmt.pix.width = camera_width;
  fmt.fmt.pix.height = camera_height;
  // jpeg
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
  fmt.fmt.pix.field = V4L2_FIELD_ANY;
  if (0 != ioctl(cam_fd_, VIDIOC_S_FMT, &fmt))
    SPDLOG_CRITICAL("VIDIOC_S_FMT");
  SPDLOG_INFO("{},{} sizeimage: {}", fmt.fmt.pix.width, fmt.fmt.pix.height, fmt.fmt.pix.sizeimage);

  // set fps
  struct v4l2_streamparm streamparm;
  memset(&streamparm, 0, sizeof(struct v4l2_streamparm));
  streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  streamparm.parm.capture.timeperframe.numerator = 1;
  streamparm.parm.capture.timeperframe.denominator = config_["camera"]["fps"].as<unsigned int>();
  if (0 != ioctl(cam_fd_, VIDIOC_G_PARM, &streamparm))
    SPDLOG_CRITICAL("VIDIOC_G_PARM");
  SPDLOG_INFO("FPS: {} / {}",
              streamparm.parm.capture.timeperframe.denominator,
              streamparm.parm.capture.timeperframe.numerator);

  // using mmap instead of dmabuf
  struct v4l2_requestbuffers reqbuf;
  memset(&reqbuf, 0, sizeof(reqbuf));
  reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  reqbuf.memory = V4L2_MEMORY_MMAP;
  reqbuf.count = 1;
  if (ioctl(cam_fd_, VIDIOC_REQBUFS, &reqbuf) == -1)
    SPDLOG_CRITICAL("VIDIOC_REQBUFS");

  // allocate memory
  // jpeg for gRPC (CPU address)
  for (size_t i = 0; i < mjpeg_num; i++) {
    encode_jpeg_list[i].start = (unsigned char*)malloc(encode_jpeg_max_size_);
  }
  // init detector_fd & depth_fd
  NvBufferCreateParams input_params = {0};
  input_params.payloadType = NvBufferPayload_SurfArray;
  input_params.width = model_size;
  input_params.height = model_size;
  input_params.layout = NvBufferLayout_Pitch;
  input_params.colorFormat = NvBufferColorFormat_ABGR32;
  input_params.nvbuf_tag = NvBufferTag_NONE;
  if (0 != NvBufferCreateEx(&detector_fd, &input_params))
    SPDLOG_CRITICAL("NvBufferCreateEx");
  input_params.width = depth_size;
  input_params.height = depth_size;
  if (0 != NvBufferCreateEx(&depth_fd, &input_params))
    SPDLOG_CRITICAL("NvBufferCreateEx");

  // NvBufferTransformParams, FILTER & CROP
  memset(&transform_params_, 0, sizeof(NvBufferTransformParams));
  transform_params_.transform_flag = NVBUFFER_TRANSFORM_FILTER;
  transform_params_.transform_filter = NvBufferTransform_Filter_Nicest;

  // queue buff in v4l2
  struct v4l2_buffer buf;
  memset(&buf, 0, sizeof(buf));
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  buf.index = 0;
  if (ioctl(cam_fd_, VIDIOC_QUERYBUF, &buf) < 0)
    SPDLOG_CRITICAL("VIDIOC_QUERYBUF");
  capture_buffer_.size = buf.length;
  // mmap
  capture_buffer_.start = (unsigned char*)mmap(NULL /* start anywhere */,
                                               buf.length, PROT_READ | PROT_WRITE /* required */,
                                               MAP_SHARED /* recommended */,
                                               cam_fd_, buf.m.offset);
  if (MAP_FAILED == capture_buffer_.start)
    SPDLOG_CRITICAL("Failed to map buffers");
  if (0 != ioctl(cam_fd_, VIDIOC_QBUF, &buf))
    SPDLOG_CRITICAL("VIDIOC_QBUF");

  // start_capturing
  enum v4l2_buf_type type;
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (0 != ioctl(cam_fd_, VIDIOC_STREAMON, &type))
    SPDLOG_CRITICAL("VIDIOC_STREAMON");

  // warm up
  usleep(1000000);  // wait for hardware
  Capture();
  Encode();
  return 0;
}

int Camera::Capture() {
  SPDLOG_DEBUG("Strat");
  // poll event
  struct pollfd fds[1];
  fds[0].fd = cam_fd_;
  fds[0].events = POLLIN;
  SPDLOG_TRACE("poll");
  if (-1 == poll(fds, 1, 100))
    SPDLOG_CRITICAL("poll");
  if (!(fds[0].revents & POLLIN))
    SPDLOG_CRITICAL("revents");

  SPDLOG_TRACE("v4l2_buffer");
  /* Dequeue a camera buff */
  struct v4l2_buffer v4l2_buf;
  memset(&v4l2_buf, 0, sizeof(v4l2_buf));
  v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  v4l2_buf.memory = V4L2_MEMORY_MMAP;
  SPDLOG_TRACE("VIDIOC_DQBUF");
  if (0 != ioctl(cam_fd_, VIDIOC_DQBUF, &v4l2_buf))
    SPDLOG_CRITICAL("VIDIOC_DQBUF");

  SPDLOG_TRACE("bytesused");
  /* v4l2_buf.bytesused may have padding bytes for alignment
     Search for EOF to get exact size */
  unsigned int bytesused = v4l2_buf.bytesused;
  uint8_t* p;
  while (bytesused) {
    p = (uint8_t*)(capture_buffer_.start + bytesused);
    if ((*(p - 2) == 0xff) && (*(p - 1) == 0xd9)) {
      break;
    }
    bytesused--;
  }

  SPDLOG_TRACE("Encode");
  // switch buffer index
  encode_index = 1 - encode_index;
  encode_jpeg_list[encode_index].size = bytesused;
  // copy the jpeg in to buffer
  // I don't need to encode it, since the data from camera is already jpeg
  memcpy(encode_jpeg_list[encode_index].start, capture_buffer_.start, bytesused);

  // decode jpeg
  SPDLOG_TRACE("decodeToFd");
  uint32_t width, height, pixfmt;
  int ret = jpegdec_->decodeToFd(capture_buffer_.dmabuff_fd, capture_buffer_.start, bytesused, pixfmt, width, height);
  if (ret < 0)
    SPDLOG_CRITICAL("decodeToFd");
  // check the format
  if (pixfmt != V4L2_PIX_FMT_YUV422M)
    SPDLOG_WARN("pixfmt != V4L2_PIX_FMT_YUV422M");

  // convert YUV422 -> rgba
  SPDLOG_TRACE("NvBufferTransform"); 
  if (0 != NvBufferTransform(capture_buffer_.dmabuff_fd, detector_fd, &transform_params_))
    SPDLOG_CRITICAL("Failed to convert the buffer");

  // convert size, detector size -> depth size
  SPDLOG_TRACE("NvBufferTransform");  
  if (0 != NvBufferTransform(detector_fd, depth_fd, &transform_params_))
    SPDLOG_CRITICAL("Failed to convert the buffer");

  // queue the buffer again
  SPDLOG_TRACE("VIDIOC_QBUF");
  if (0 != ioctl(cam_fd_, VIDIOC_QBUF, &v4l2_buf))
    SPDLOG_CRITICAL("VIDIOC_QBUF");
  SPDLOG_TRACE("End");
  return 0;
}

// I don't need to encode images, since the data from camera is already jpeg
int Camera::Encode() {
  // SPDLOG_TRACE("Strat");
  // SPDLOG_TRACE("Encode");
  // encode_index = 1 - encode_index;
  // encode_jpeg_list[encode_index].size = encode_jpeg_max_size_;
  // if (0 != jpegenc_->encodeFromFd(capture_fd, JCS_YCbCr,
  //                                 &encode_jpeg_list[encode_index].start, encode_jpeg_list[encode_index].size,
  //                                 encode_quality_))
  //   SPDLOG_CRITICAL("encodeFromFd");
  // if (encode_jpeg_list[encode_index].size > encode_jpeg_max_size_)
  //   SPDLOG_CRITICAL("encode_jpeg_max_size_ < {}", encode_jpeg_list[encode_index].size);
  // SPDLOG_TRACE("End");
  return 0;
}

Camera::~Camera() {
  // stop_capturing
  enum v4l2_buf_type type;
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (0 != ioctl(cam_fd_, VIDIOC_STREAMOFF, &type))
    SPDLOG_CRITICAL("VIDIOC_STREAMOFF");
  // uninit_device
  munmap(capture_buffer_.start, capture_buffer_.size);
  // close_device
  if (0 != close(cam_fd_))
    SPDLOG_CRITICAL("close");
  // free memory
  for (size_t i = 0; i < mjpeg_num; i++) {
    free(encode_jpeg_list[i].start);
  }
  delete jpegdec_;
  delete jpegenc_;
}

}  // namespace jetson