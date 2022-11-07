#include "camera.h"

#include "model.h"

// Ref: /usr/src/jetson_multimedia_api/samples/

#include <asm/types.h> /* for videodev2.h */
#include <fcntl.h>
#include <jetson-utils/cudaColorspace.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/cudaNormalize.h>
#include <jetson-utils/cudaResize.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "cudaEGL.h"

namespace jetson {

Camera::Camera(YAML::Node& config, Model& model) : config_(config), model_(model) {
  model_size_ = config_["detector"]["size"].as<unsigned int>();
  memset(out_jpeg_buffer, 0, mjpeg_num * sizeof(nv_buffer));
}

int Camera::Open() {
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

  struct v4l2_streamparm streamparm;
  memset(&streamparm, 0, sizeof(struct v4l2_streamparm));
  streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  // try to set fps. But it does not work.
  streamparm.parm.capture.timeperframe.numerator = 1;
  streamparm.parm.capture.timeperframe.denominator = 30;
  if (-1 == ioctl(cam_fd_, VIDIOC_G_PARM, &streamparm))
    SPDLOG_CRITICAL("VIDIOC_G_PARM");

  // mmap instead of dmabuf for mjpeg
  struct v4l2_requestbuffers reqbuf;
  memset(&reqbuf, 0, sizeof(reqbuf));
  reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  reqbuf.memory = V4L2_MEMORY_MMAP;
  reqbuf.count = 1;
  if (ioctl(cam_fd_, VIDIOC_REQBUFS, &reqbuf) == -1)
    SPDLOG_CRITICAL("VIDIOC_REQBUFS");

  // allocate memory
  // out_jpeg_buffer for gRPC in CPU address
  for (size_t i = 0; i < mjpeg_num; i++) {
    // how to set the maximal size?
    out_jpeg_buffer->start = (unsigned char*)malloc(model_size_ * model_size_ * 3);
  }
  // trans_buffer for network in GPU fd
  NvBufferCreateParams input_params = {0};
  input_params.payloadType = NvBufferPayload_SurfArray;
  input_params.width = config_["camera"]["width"].as<unsigned int>() / 2;
  input_params.height = config_["camera"]["height"].as<unsigned int>();
  input_params.layout = NvBufferLayout_Pitch;
  input_params.colorFormat = NvBufferColorFormat_ABGR32;
  input_params.nvbuf_tag = NvBufferTag_CAMERA;
  if (-1 == NvBufferCreateEx(&trans_buffer_.dmabuff_fd, &input_params))
    SPDLOG_CRITICAL("NvBufferCreateEx");
  // NvBufferTransformParams, FILTER & CROP
  memset(&transParams_, 0, sizeof(transParams_));
  transParams_.transform_flag = NVBUFFER_TRANSFORM_CROP_SRC | NVBUFFER_TRANSFORM_FILTER;
  transParams_.transform_filter = NvBufferTransform_Filter_Smart;
  transParams_.src_rect.top = 0;
  transParams_.src_rect.left = 0;
  transParams_.src_rect.width = input_params.width;
  transParams_.src_rect.height = input_params.height;

  /* Create egl_display that will be used in mapping DMABUF to CUDA buffer */
  egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (egl_display_ == EGL_NO_DISPLAY)
    SPDLOG_CRITICAL("Error while get EGL display connection");
  if (!eglInitialize(egl_display_, NULL, NULL))
    SPDLOG_CRITICAL("Erro while initialize EGL display connection");
  if (!cudaAllocMapped((void**)&rgb_image_, config_["camera"]["width"].as<unsigned int>(),
                       config_["camera"]["height"].as<unsigned int>(), IMAGE_RGB32F))
    SPDLOG_CRITICAL("cudaAllocMapped");
  if (!cudaAllocMapped((void**)&resize_image_, model_size_, model_size_, IMAGE_RGB32F))
    SPDLOG_CRITICAL("cudaAllocMapped");
  if (!cudaAllocMapped((void**)&model_image_, model_size_, model_size_, IMAGE_RGB32F))
    SPDLOG_CRITICAL("cudaAllocMapped");
  // model_image_ = (float3*)model_.GetInputPtr();

  // queue buff
  struct v4l2_buffer buf;
  memset(&buf, 0, sizeof(buf));
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  buf.index = 0;
  if (ioctl(cam_fd_, VIDIOC_QUERYBUF, &buf) < 0)
    SPDLOG_CRITICAL("VIDIOC_QUERYBUF");
  in_jpeg_buffer_.size = buf.length;
  in_jpeg_buffer_.start = (unsigned char*)
      mmap(NULL /* start anywhere */,
           buf.length,
           PROT_READ | PROT_WRITE /* required */,
           MAP_SHARED /* recommended */,
           cam_fd_, buf.m.offset);
  if (MAP_FAILED == in_jpeg_buffer_.start)
    SPDLOG_CRITICAL("Failed to map buffers");
  if (-1 == ioctl(cam_fd_, VIDIOC_QBUF, &buf))
    SPDLOG_CRITICAL("VIDIOC_QBUF");
  // start_capturing
  enum v4l2_buf_type type;
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == ioctl(cam_fd_, VIDIOC_STREAMON, &type))
    SPDLOG_CRITICAL("VIDIOC_STREAMON");
  // warm up
  usleep(5000000);  // wait for hardware?
  Capture();
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
  v4l2_buf.memory = V4L2_MEMORY_MMAP;
  SPDLOG_TRACE("VIDIOC_DQBUF");
  if (-1 == ioctl(cam_fd_, VIDIOC_DQBUF, &v4l2_buf))
    SPDLOG_CRITICAL("VIDIOC_DQBUF");

  SPDLOG_TRACE("bytesused");
  /* v4l2_buf.bytesused may have padding bytes for alignment
     Search for EOF to get exact size */
  unsigned int bytesused = v4l2_buf.bytesused;
  uint8_t* p;
  while (bytesused) {
    p = (uint8_t*)(in_jpeg_buffer_.start + bytesused);
    if ((*(p - 2) == 0xff) && (*(p - 1) == 0xd9)) {
      break;
    }
    bytesused--;
  }

  SPDLOG_TRACE("decodeToFd");
  int fd = 0;
  uint32_t width, height, pixfmt;
  int ret = jpegdec_->decodeToFd(fd, in_jpeg_buffer_.start, bytesused, pixfmt, width, height);
  if (ret < 0)
    SPDLOG_CRITICAL("decodeToFd");
  if (pixfmt != V4L2_PIX_FMT_YUV422M)
    SPDLOG_WARN("pixfmt != V4L2_PIX_FMT_YUV422M");

  SPDLOG_TRACE("VIDIOC_QBUF");
  if (-1 == ioctl(cam_fd_, VIDIOC_QBUF, &v4l2_buf))
    SPDLOG_CRITICAL("VIDIOC_QBUF");
  SPDLOG_TRACE("Convert");
  Convert(fd);
  SPDLOG_TRACE("End");
  return 0;
}

int Camera::Convert(int fd) {
  SPDLOG_TRACE("Strat");

  if (-1 == NvBufferTransform(fd, trans_buffer_.dmabuff_fd, &transParams_))
    SPDLOG_CRITICAL("Failed to convert the buffer");

  SPDLOG_TRACE("Read Fd");
  EGLImageKHR egl_image = NvEGLImageFromFd(egl_display_, trans_buffer_.dmabuff_fd);
  if (egl_image == NULL)
    SPDLOG_CRITICAL("NvEGLImageFromFd");
  CUresult status;
  CUeglFrame eglFrame;
  CUgraphicsResource pResource = NULL;
  cudaFree(0);
  status = cuGraphicsEGLRegisterImage(&pResource, egl_image,
                                      CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
  if (status != CUDA_SUCCESS)
    SPDLOG_CRITICAL("cuGraphicsEGLRegisterImage");
  status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
  if (status != CUDA_SUCCESS)
    SPDLOG_CRITICAL("cuGraphicsResourceGetMappedEglFrame");
  status = cuCtxSynchronize();
  if (status != CUDA_SUCCESS)
    SPDLOG_CRITICAL("cuCtxSynchronize");
  if (eglFrame.frameType != CU_EGL_FRAME_TYPE_PITCH)
    SPDLOG_CRITICAL("CU_EGL_FRAME_TYPE_PITCH");

  SPDLOG_TRACE("jpeg");

  SPDLOG_TRACE("Model");
  model_.Process((void*)eglFrame.frame.pArray[0]);

  SPDLOG_TRACE("Synchronize");
  cudaDeviceSynchronize();
  // SPDLOG_TRACE("cuCtxSynchronize");
  // status = cuCtxSynchronize();
  // if (status != CUDA_SUCCESS)
  //   SPDLOG_CRITICAL("cuCtxSynchronize");
  SPDLOG_TRACE("cleanup");
  status = cuGraphicsUnregisterResource(pResource);
  if (status != CUDA_SUCCESS)
    SPDLOG_CRITICAL("cuGraphicsUnregisterResource");
  NvDestroyEGLImage(egl_display_, egl_image);
  SPDLOG_TRACE("End");
  return 0;
}

Camera::~Camera() {
  //  stop_capturing
  enum v4l2_buf_type type;
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == ioctl(cam_fd_, VIDIOC_STREAMOFF, &type))
    SPDLOG_CRITICAL("VIDIOC_STREAMOFF");
  // uninit_device
  munmap(in_jpeg_buffer_.start, in_jpeg_buffer_.size);
  NvBufferDestroy(trans_buffer_.dmabuff_fd);
  // close_device
  if (-1 == close(cam_fd_))
    SPDLOG_CRITICAL("close");
  if (!eglTerminate(egl_display_))
    SPDLOG_CRITICAL("eglTerminate");
  // free memory
  cudaFreeHost(rgb_image_);
  cudaFreeHost(resize_image_);
  // cudaFreeHost(model_image_);
  delete jpegdec_;
}

}  // namespace jetson