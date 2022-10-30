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
#include <unistd.h>

#include "cudaEGL.h"

namespace jetson {

Camera::Camera(YAML::Node& config, Model& model) : config_(config), model_(model) {
  model_size_ = config_["detector"]["size"].as<unsigned int>();
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
  input_params.payloadType = NvBufferPayload_SurfArray;
  input_params.width = config_["camera"]["width"].as<unsigned int>();
  input_params.height = config_["camera"]["height"].as<unsigned int>();
  input_params.layout = NvBufferLayout_Pitch;
  input_params.colorFormat = NvBufferColorFormat_ABGR32;
  input_params.nvbuf_tag = NvBufferTag_CAMERA;
  if (-1 == NvBufferCreateEx(&trans_buffer_.dmabuff_fd, &input_params))
    SPDLOG_CRITICAL("NvBufferCreateEx");
  if (-1 == NvBufferMemMap(trans_buffer_.dmabuff_fd, 0, NvBufferMem_Read_Write,
                           (void**)&trans_buffer_.start))
    SPDLOG_CRITICAL("NvBufferMemMap");
  // NvBufferTransformParams
  memset(&transParams_, 0, sizeof(transParams_));
  transParams_.transform_flag = NVBUFFER_TRANSFORM_FILTER;
  transParams_.transform_filter = NvBufferTransform_Filter_Smart;
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
  // NvBufferMemSyncForDevice(mjpeg_buffer[mjpeg_index].dmabuff_fd, 0,
  //                          (void**)&mjpeg_buffer[mjpeg_index].start);
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
  SPDLOG_TRACE("Convert");
  Convert();
  SPDLOG_TRACE("End");
  return 0;
}

int Camera::Convert() {
  SPDLOG_TRACE("Strat");

  SPDLOG_TRACE("decodeToFd");
  int fd = 0;
  uint32_t width, height, pixfmt;
  int ret = jpegdec_->decodeToFd(fd, mjpeg_buffer[(mjpeg_index + mjpeg_num - 1) % mjpeg_num].start, mjpeg_size, pixfmt, width, height);
  if (ret < 0)
    SPDLOG_CRITICAL("decodeToFd");
  if (pixfmt != V4L2_PIX_FMT_YUV420M)
    SPDLOG_WARN("pixfmt != V4L2_PIX_FMT_YUV420M");

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
  for (int i = 0; i < mjpeg_num; i++)
    NvBufferDestroy(mjpeg_buffer[i].dmabuff_fd);
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