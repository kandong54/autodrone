#include "camera.h"

#include "model.h"

// Ref: /usr/src/jetson_multimedia_api/samples/
//      https://github.com/dusty-nv/jetson-utils/blob/master/codec/gstBufferManager.cpp

#include <asm/types.h> /* for videodev2.h */
#include <cuda_egl_interop.h>
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

namespace jetson {

Camera::Camera(YAML::Node& config, Model& model) : config_(config), model_(model) {
  memset(encode_jpeg_list, 0, mjpeg_num * sizeof(nv_buffer));
  const int model_size = config_["detector"]["size"].as<unsigned int>();
  // how to set the maximal size?
  encode_jpeg_max_size_ = model_size * model_size * 3;
  encode_quality_ = config_["camera"]["quality"].as<unsigned int>();
}

int Camera::Open() {
  // const
  const int model_size = config_["detector"]["size"].as<unsigned int>();
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
  fmt.fmt.pix.width = camera_width;
  fmt.fmt.pix.height = camera_height;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
  fmt.fmt.pix.field = V4L2_FIELD_ANY;
  if (-1 == ioctl(cam_fd_, VIDIOC_S_FMT, &fmt))
    SPDLOG_CRITICAL("VIDIOC_S_FMT");
  SPDLOG_INFO("sizeimage: {}", fmt.fmt.pix.sizeimage);

  struct v4l2_streamparm streamparm;
  memset(&streamparm, 0, sizeof(struct v4l2_streamparm));
  streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  // try to set fps. But it does not work.
  streamparm.parm.capture.timeperframe.numerator = 1;
  streamparm.parm.capture.timeperframe.denominator = 30;
  if (-1 == ioctl(cam_fd_, VIDIOC_G_PARM, &streamparm))
    SPDLOG_CRITICAL("VIDIOC_G_PARM");
  SPDLOG_INFO("FPS: {} / {}",
              streamparm.parm.capture.timeperframe.denominator,
              streamparm.parm.capture.timeperframe.numerator);

  // mmap instead of dmabuf
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
  // detect_rbg_buffer_
  NvBufferCreateParams input_params = {0};
  input_params.payloadType = NvBufferPayload_SurfArray;
  input_params.width = config_["camera"]["width"].as<unsigned int>() / 2;
  input_params.height = config_["camera"]["height"].as<unsigned int>();
  input_params.layout = NvBufferLayout_Pitch;
  input_params.colorFormat = NvBufferColorFormat_ABGR32;
  input_params.nvbuf_tag = NvBufferTag_NONE;
  if (-1 == NvBufferCreateEx(&detect_rbg_buffer_.dmabuff_fd, &input_params))
    SPDLOG_CRITICAL("NvBufferCreateEx");
  // encode_yuv_buffer_
  input_params.colorFormat = NvBufferColorFormat_YUV420;
  input_params.nvbuf_tag = NvBufferTag_JPEG;
  if (-1 == NvBufferCreateEx(&encode_yuv_buffer_.dmabuff_fd, &input_params))
    SPDLOG_CRITICAL("NvBufferCreateEx");
  // transParams_, FILTER & CROP
  memset(&transParams_, 0, sizeof(NvBufferTransformParams));
  transParams_.transform_flag = NVBUFFER_TRANSFORM_CROP_SRC | NVBUFFER_TRANSFORM_FILTER;
  transParams_.transform_filter = NvBufferTransform_Filter_Smart;
  transParams_.src_rect.top = 0;
  transParams_.src_rect.left = 0;
  transParams_.src_rect.width = input_params.width;
  transParams_.src_rect.height = input_params.height;

  // queue buff
  struct v4l2_buffer buf;
  memset(&buf, 0, sizeof(buf));
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  buf.index = 0;
  if (ioctl(cam_fd_, VIDIOC_QUERYBUF, &buf) < 0)
    SPDLOG_CRITICAL("VIDIOC_QUERYBUF");
  capture_jpeg_buffer_.size = buf.length;
  capture_jpeg_buffer_.start = (unsigned char*)
      mmap(NULL /* start anywhere */,
           buf.length,
           PROT_READ | PROT_WRITE /* required */,
           MAP_SHARED /* recommended */,
           cam_fd_, buf.m.offset);
  if (MAP_FAILED == capture_jpeg_buffer_.start)
    SPDLOG_CRITICAL("Failed to map buffers");
  if (-1 == ioctl(cam_fd_, VIDIOC_QBUF, &buf))
    SPDLOG_CRITICAL("VIDIOC_QBUF");
  // start_capturing
  enum v4l2_buf_type type;
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == ioctl(cam_fd_, VIDIOC_STREAMON, &type))
    SPDLOG_CRITICAL("VIDIOC_STREAMON");
  // warm up
  usleep(3000000);  // wait for hardware?
  Capture();        // skip the first frame
  Capture();
  Encode();
  Depth();
  Detect();
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
    p = (uint8_t*)(capture_jpeg_buffer_.start + bytesused);
    if ((*(p - 2) == 0xff) && (*(p - 1) == 0xd9)) {
      break;
    }
    bytesused--;
  }

  SPDLOG_TRACE("decodeToFd");
  uint32_t width, height, pixfmt;
  int ret = jpegdec_->decodeToFd(capture_yuv_fd_, capture_jpeg_buffer_.start, bytesused, pixfmt, width, height);
  if (ret < 0)
    SPDLOG_CRITICAL("decodeToFd");
  if (pixfmt != V4L2_PIX_FMT_YUV422M)
    SPDLOG_WARN("pixfmt != V4L2_PIX_FMT_YUV422M");

  cudaDeviceSynchronize();

  SPDLOG_TRACE("VIDIOC_QBUF");
  if (-1 == ioctl(cam_fd_, VIDIOC_QBUF, &v4l2_buf))
    SPDLOG_CRITICAL("VIDIOC_QBUF");
  SPDLOG_TRACE("End");
  return 0;
}

int Camera::Encode() {
  SPDLOG_TRACE("Strat");
  SPDLOG_TRACE("NvBufferTransform");  // YUV422 & stereo -> YUV420 & right
  if (-1 == NvBufferTransform(capture_yuv_fd_, encode_yuv_buffer_.dmabuff_fd, &transParams_))
    SPDLOG_CRITICAL("Failed to convert the buffer");
  SPDLOG_TRACE("Encode");
  encode_index = 1 - encode_index;
  encode_jpeg_list[encode_index].size = encode_jpeg_max_size_;
  if (0 != jpegenc_->encodeFromFd(encode_yuv_buffer_.dmabuff_fd, JCS_YCbCr,
                                  &encode_jpeg_list[encode_index].start, encode_jpeg_list[encode_index].size,
                                  encode_quality_))
    SPDLOG_CRITICAL("encodeFromFd");
  if (encode_jpeg_list[encode_index].size > encode_jpeg_max_size_)
    SPDLOG_CRITICAL("encode_jpeg_max_size_ < {}", encode_jpeg_list[encode_index].size);
  SPDLOG_TRACE("End");
  return 0;
}

int Camera::Depth() {
  SPDLOG_TRACE("Strat");

  SPDLOG_TRACE("End");
  return 0;
}

int Camera::Detect() {
  SPDLOG_TRACE("Strat");

  SPDLOG_TRACE("NvBufferTransform");  // YUV422 & stereo -> rgb & right
  if (-1 == NvBufferTransform(capture_yuv_fd_, detect_rbg_buffer_.dmabuff_fd, &transParams_))
    SPDLOG_CRITICAL("Failed to convert the buffer");

  // uchar4* pdata;
  // NvBufferMemMap(detect_rbg_buffer_.dmabuff_fd, 0, NvBufferMem_Read, (void**)&pdata);
  // NvBufferMemSyncForCpu(detect_rbg_buffer_.dmabuff_fd, 0, (void**)&pdata);
  // NvBufferMemUnMap(detect_rbg_buffer_.dmabuff_fd, 0, (void**)&pdata);

  SPDLOG_TRACE("Read Fd");
  EGLImageKHR egl_image = NvEGLImageFromFd(NULL, detect_rbg_buffer_.dmabuff_fd);
  if (egl_image == NULL)
    SPDLOG_CRITICAL("NvEGLImageFromFd");
  cudaEglFrame eglFrame;
  cudaGraphicsResource* eglResource = NULL;
  //cudaFree(0);
  if (CUDA_FAILED(cudaGraphicsEGLRegisterImage(&eglResource, egl_image,
                                               CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)))
    SPDLOG_CRITICAL("cudaGraphicsEGLRegisterImage");
  if (CUDA_FAILED(cudaGraphicsResourceGetMappedEglFrame(&eglFrame, eglResource, 0, 0)))
    SPDLOG_CRITICAL("cuGraphicsResourceGetMappedEglFrame");

  if (eglFrame.frameType != cudaEglFrameTypePitch)
    SPDLOG_CRITICAL("{} != cudaEglFrameTypePitch", eglFrame.frameType);

  SPDLOG_TRACE("Model");
  model_.Process((void*)eglFrame.frame.pArray[0]);
  // uchar4 tmp[2560/2][720];
  // cudaMemcpy(tmp, eglFrame.frame.pArray[0],2560/2 * 720 * 4, cudaMemcpyDeviceToHost);

  SPDLOG_TRACE("Synchronize");
  cudaDeviceSynchronize();
  SPDLOG_TRACE("cleanup");
  if (CUDA_FAILED(cudaGraphicsUnregisterResource(eglResource)))
    SPDLOG_CRITICAL("cudaGraphicsUnregisterResource");
  NvDestroyEGLImage(NULL, egl_image);
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
  munmap(capture_jpeg_buffer_.start, capture_jpeg_buffer_.size);
  // close_device
  if (-1 == close(cam_fd_))
    SPDLOG_CRITICAL("close");
  // free memory
  NvBufferDestroy(detect_rbg_buffer_.dmabuff_fd);
  NvBufferDestroy(encode_yuv_buffer_.dmabuff_fd);
  for (size_t i = 0; i < mjpeg_num; i++) {
    free(encode_jpeg_list[i].start);
  }
  delete jpegdec_;
  delete jpegenc_;
}

}  // namespace jetson