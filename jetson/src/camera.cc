#include "camera.h"
// /usr/src/jetson_multimedia_api/samples/v4l2cuda/capture.cpp
// /usr/src/jetson_multimedia_api/samples/12_camera_v4l2_cuda/camera_v4l2_cuda.cpp

#include <asm/types.h> /* for videodev2.h */
#include <cuda_runtime.h>
#include <fcntl.h>
#include <jetson-utils/nvbuf_utils.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace jetson {

Camera::Camera(YAML::Node& config) : config_(config) {
}

int Camera::Open() {
  // open_device
  std::string dev_name = config_["camera"]["device"].as<std::string>();
  cam_fd_ = open(dev_name.c_str(), O_RDWR /* required */ | O_NONBLOCK, 0);
  if (-1 == cam_fd_)
    SPDLOG_CRITICAL("Failed to open {}", dev_name);

  // init_device
  struct v4l2_capability cap;
  ioctl(cam_fd_, VIDIOC_QUERYCAP, &cap);
  if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)
    SPDLOG_INFO("V4L2_CAP_VIDEO_CAPTURE");
  if (cap.capabilities & V4L2_CAP_READWRITE)
    SPDLOG_INFO("V4L2_CAP_READWRITE");
  if (cap.capabilities & V4L2_CAP_ASYNCIO)
    SPDLOG_INFO("V4L2_CAP_ASYNCIO");
  if (cap.capabilities & V4L2_CAP_STREAMING)
    SPDLOG_INFO("V4L2_CAP_STREAMING");

  struct v4l2_format fmt;
  memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  // v4l2-ctl --list-formats-ext
  fmt.fmt.pix.width = config_["camera"]["width"].as<unsigned int>();
  fmt.fmt.pix.height = config_["camera"]["height"].as<unsigned int>();
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
  fmt.fmt.pix.field = V4L2_FIELD_ANY;
  if (-1 == ioctl(cam_fd_, VIDIOC_S_FMT, &fmt))
    SPDLOG_CRITICAL("VIDIOC_S_FMT");
  sizeimage_ = fmt.fmt.pix.sizeimage;

  struct v4l2_requestbuffers reqbuf;
  memset(&reqbuf, 0, sizeof(reqbuf));
  reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  reqbuf.memory = V4L2_MEMORY_DMABUF;
  reqbuf.count = 1;
  if (ioctl(cam_fd_, VIDIOC_REQBUFS, &reqbuf) == -1)
    SPDLOG_CRITICAL("VIDIOC_REQBUFS");
  // init_cuda
  NvBufferCreateParams input_params = {0};
  input_params.payloadType = NvBufferPayload_MemHandle;
  input_params.memsize = sizeimage_;
  input_params.nvbuf_tag = NvBufferTag_CAMERA;
  for (int i = 0; i < mjpeg_num; i++) {
    if (-1 == NvBufferCreateEx(&mjpeg_buffer[i].dmabuff_fd, &input_params))
      SPDLOG_CRITICAL("NvBufferCreateEx");
    if (-1 == NvBufferMemMap(mjpeg_buffer[i].dmabuff_fd, 0, NvBufferMem_Read_Write,
                             (void**)&mjpeg_buffer[i].start))
      SPDLOG_CRITICAL("NvBufferMemMap");
  }
  // start_capturing
  struct v4l2_buffer buf;
  memset(&buf, 0, sizeof(buf));
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_DMABUF;
  buf.index = 0;
  if (ioctl(cam_fd_, VIDIOC_QUERYBUF, &buf) < 0)
    SPDLOG_CRITICAL("VIDIOC_QUERYBUF");
  mjpeg_index = 0;
  buf.m.fd = (unsigned long)mjpeg_buffer[mjpeg_index].dmabuff_fd;
  if (-1 == ioctl(cam_fd_, VIDIOC_QBUF, &buf))
    SPDLOG_CRITICAL("VIDIOC_QBUF");
  enum v4l2_buf_type type;
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == ioctl(cam_fd_, VIDIOC_STREAMON, &type))
    SPDLOG_CRITICAL("VIDIOC_STREAMON");
  // warm up
  Capture();
  Transfer();
  return 0;
}

int Camera::Capture() {
  SPDLOG_DEBUG("Strat");
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
  v4l2_buf.memory = V4L2_MEMORY_DMABUF;
  SPDLOG_TRACE("VIDIOC_DQBUF");
  if (-1 == ioctl(cam_fd_, VIDIOC_DQBUF, &v4l2_buf))
    SPDLOG_CRITICAL("VIDIOC_DQBUF");
  SPDLOG_TRACE("NvBufferMemSyncForDevice");
  /* Cache sync for VIC operation since the data is from CPU */
  NvBufferMemSyncForDevice(mjpeg_buffer[mjpeg_index].dmabuff_fd, 0,
                           (void**)&mjpeg_buffer[mjpeg_index].start);
  SPDLOG_TRACE("bytesused");
  /* v4l2_buf.bytesused may have padding bytes for alignment
     Search for EOF to get exact size */
  unsigned int bytesused = v4l2_buf.bytesused;
  uint8_t* p;
  while (bytesused) {
    p = (uint8_t*)(mjpeg_buffer[mjpeg_index].start + bytesused);
    if ((*(p - 2) == 0xff) && (*(p - 1) == 0xd9)) {
      break;
    }
    bytesused--;
  }
  // update index
  mjpeg_size = bytesused;
  mjpeg_index = (mjpeg_index + 1) % mjpeg_num;
  SPDLOG_TRACE("VIDIOC_QBUF");
  v4l2_buf.m.fd = (unsigned long)mjpeg_buffer[mjpeg_index].dmabuff_fd;
  if (-1 == ioctl(cam_fd_, VIDIOC_QBUF, &v4l2_buf))
    SPDLOG_CRITICAL("VIDIOC_QBUF");
  SPDLOG_TRACE("End");
  return 0;
}

int Camera::Transfer() {
  SPDLOG_TRACE("Strat");
  // FILE* fp = fopen("file_name.jpg", "wb");
  // fwrite(mjpeg_buffer[0].start, 1, mjpeg_size, fp);
  // fclose(fp);
  SPDLOG_TRACE("End");

  // /* Decoding MJPEG frame */
  // if (ctx->jpegdec->decodeToFd(fd, ctx->g_buff[v4l2_buf.index].start,
  //                              bytesused, pixfmt, width, height) < 0)
  //   ERROR_RETURN("Cannot decode MJPEG");

  // /* Convert the decoded buffer to YUV420P */
  // if (-1 == NvBufferTransform(fd, ctx->render_dmabuf_fd,
  //                             &transParams))
  //   ERROR_RETURN("Failed to convert the buffer");
  return 0;
}

bool Camera::IsOpened() {
}

Camera::~Camera() {
  //  stop_capturing
  enum v4l2_buf_type type;
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == ioctl(cam_fd_, VIDIOC_STREAMOFF, &type))
    SPDLOG_CRITICAL("VIDIOC_STREAMOFF");
  // uninit_device
  for (int i = 0; i < mjpeg_num; i++)
    NvBufferDestroy(mjpeg_buffer[i].dmabuff_fd);
  // close_device
  if (-1 == close(cam_fd_))
    SPDLOG_CRITICAL("close");
}

}  // namespace jetson