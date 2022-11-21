#include "camera.h"

#include "model.h"

// Ref: /usr/src/jetson_multimedia_api/samples/
//      https://github.com/dusty-nv/jetson-utils/blob/master/codec/gstBufferManager.cpp
//      https://github.com/NVIDIA-AI-IOT/jetson-stereo-depth/blob/master/detph_pipeline_cpp/main.cpp

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
#include <vpi/CUDAInterop.h>
#include <vpi/NvBufferInterop.h>
#include <vpi/algo/StereoDisparity.h>

#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>

namespace jetson {

Camera::Camera(YAML::Node& config, Model& model) : config_(config), model_(model) {
  memset(encode_jpeg_list, 0, mjpeg_num * sizeof(nv_buffer));
  const int model_size = config_["detector"]["size"].as<unsigned int>();
  // how to set the maximal size?
  encode_jpeg_max_size_ = model_size * model_size * 3;
  encode_quality_ = config_["camera"]["quality"].as<unsigned int>();
  depth_factor_ = config_["detector"]["depth_factor"].as<unsigned int>();
  camera_width_ = config_["camera"]["width"].as<unsigned int>() / 2;
  depth_k_ = config_["detector"]["depth_k"].as<float>();
  depth_b_ = config_["detector"]["depth_b"].as<float>();
}

int Camera::Open() {
  // const
  const int model_size = config_["detector"]["size"].as<unsigned int>();
  const int camera_width = config_["camera"]["width"].as<unsigned int>() / 2;
  const int camera_height = config_["camera"]["height"].as<unsigned int>();
  const int depth_factor = config_["detector"]["depth_factor"].as<unsigned int>();
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
  // stereo estimator
  if (0 != vpiStreamCreate(0, &depth_stream_))
    SPDLOG_CRITICAL("vpiStreamCreate");
  VPIStereoDisparityEstimatorCreationParams stereoParams;
  if (0 != vpiInitStereoDisparityEstimatorCreationParams(&stereoParams))
    SPDLOG_CRITICAL("vpiInitStereoDisparityEstimatorCreationParams");
  stereoParams.maxDisparity = 64;
  if (0 != vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, camera_width / depth_factor, camera_height / depth_factor, VPI_IMAGE_FORMAT_NV12_ER, &stereoParams, &depth_stereo_))
    SPDLOG_CRITICAL("vpiCreateStereoDisparityEstimator");
  if (0 != vpiImageCreate(camera_width / depth_factor, camera_height / depth_factor, VPI_IMAGE_FORMAT_NV12_ER, VPI_BACKEND_CUDA, &depth_left_ER_img_))
    SPDLOG_CRITICAL("vpiImageCreate");
  if (0 != vpiImageCreate(camera_width / depth_factor, camera_height / depth_factor, VPI_IMAGE_FORMAT_NV12_ER, VPI_BACKEND_CUDA, &depth_right_ER_img_))
    SPDLOG_CRITICAL("vpiImageCreate");
  if (0 != vpiImageCreate(camera_width / depth_factor, camera_height / depth_factor, VPI_IMAGE_FORMAT_NV12_ER, VPI_BACKEND_CUDA, &depth_left_nr_img_))
    SPDLOG_CRITICAL("vpiImageCreate");
  if (0 != vpiImageCreate(camera_width / depth_factor, camera_height / depth_factor, VPI_IMAGE_FORMAT_NV12_ER, VPI_BACKEND_CUDA, &depth_right_nr_img_))
    SPDLOG_CRITICAL("vpiImageCreate");
  if (0 != cudaMallocManaged(&depth_disparity_data_, (camera_width / depth_factor) * (camera_height / depth_factor) * 2))
    SPDLOG_CRITICAL("vpiImageCreate");
  if (0 != cudaMallocManaged(&depth_confidenceMap_data_, (camera_width / depth_factor) * (camera_height / depth_factor) * 2))
    SPDLOG_CRITICAL("vpiImageCreate");
  VPIImageData depth_data;
  memset(&depth_data, 0, sizeof(VPIImageData));
  depth_data.format = VPI_IMAGE_FORMAT_U16;
  depth_data.numPlanes = 1;
  depth_data.planes[0].data = depth_disparity_data_;
  depth_data.planes[0].width = camera_width / depth_factor;
  depth_data.planes[0].height = camera_height / depth_factor;
  depth_data.planes[0].pitchBytes = 2 * (camera_width / depth_factor);
  depth_data.planes[0].pixelType = VPI_PIXEL_TYPE_U16;
  if (0 != vpiImageCreateCUDAMemWrapper(&depth_data, VPI_BACKEND_CUDA, &depth_disparity_))
    SPDLOG_CRITICAL("vpiImageCreate");
  depth_data.planes[0].data = depth_confidenceMap_data_;
  if (0 != vpiImageCreateCUDAMemWrapper(&depth_data, VPI_BACKEND_CUDA, &depth_confidenceMap_))
    SPDLOG_CRITICAL("vpiImageCreate");
  if (0 != vpiInitConvertImageFormatParams(&depth_convParams_))
    SPDLOG_CRITICAL("vpiInitConvertImageFormatParams");

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
  fmt.fmt.pix.width = camera_width * 2;
  fmt.fmt.pix.height = camera_height;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
  fmt.fmt.pix.field = V4L2_FIELD_ANY;
  if (0 != ioctl(cam_fd_, VIDIOC_S_FMT, &fmt))
    SPDLOG_CRITICAL("VIDIOC_S_FMT");
  SPDLOG_INFO("{},{} sizeimage: {}", fmt.fmt.pix.width, fmt.fmt.pix.height, fmt.fmt.pix.sizeimage);

  struct v4l2_streamparm streamparm;
  memset(&streamparm, 0, sizeof(struct v4l2_streamparm));
  streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  // try to set fps. But it does not work.
  streamparm.parm.capture.timeperframe.numerator = 1;
  streamparm.parm.capture.timeperframe.denominator = 30;
  if (0 != ioctl(cam_fd_, VIDIOC_G_PARM, &streamparm))
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
  input_params.width = model_size;
  input_params.height = model_size;
  input_params.layout = NvBufferLayout_Pitch;
  input_params.colorFormat = NvBufferColorFormat_ABGR32;
  input_params.nvbuf_tag = NvBufferTag_NONE;
  if (0 != NvBufferCreateEx(&detect_rbg_fd_, &input_params))
    SPDLOG_CRITICAL("NvBufferCreateEx");
  // Depth
  input_params.width = camera_width / depth_factor;
  input_params.height = camera_height / depth_factor;
  input_params.colorFormat = NvBufferColorFormat_ABGR32;
  if (0 != NvBufferCreateEx(&depth_left_fd_, &input_params))
    SPDLOG_CRITICAL("NvBufferCreateEx");
  if (0 != NvBufferCreateEx(&depth_right_fd_, &input_params))
    SPDLOG_CRITICAL("NvBufferCreateEx");
  if (0 != vpiImageCreateNvBufferWrapper(depth_left_fd_, NULL, 0, &depth_left_img_))
    SPDLOG_CRITICAL("vpiImageCreateNvBufferWrapper");
  if (0 != vpiImageCreateNvBufferWrapper(depth_right_fd_, NULL, 0, &depth_right_img_))
    SPDLOG_CRITICAL("vpiImageCreateNvBufferWrapper");
  // denoise
  if (0 != vpiCreateTemporalNoiseReduction(VPI_BACKEND_CUDA, camera_width / depth_factor, camera_height / depth_factor, VPI_IMAGE_FORMAT_NV12_ER, VPI_TNR_DEFAULT, &depth_tnr_))
    SPDLOG_CRITICAL("vpiCreateTemporalNoiseReduction");
  vpiInitTemporalNoiseReductionParams(&depth_tnr_params_);
  depth_tnr_params_.strength = 0.5;

  // encode_yuv_buffer_
  input_params.width = camera_width;
  input_params.height = camera_height;
  input_params.colorFormat = NvBufferColorFormat_YUV420;
  input_params.nvbuf_tag = NvBufferTag_JPEG;
  if (0 != NvBufferCreateEx(&encode_yuv_fd_, &input_params))
    SPDLOG_CRITICAL("NvBufferCreateEx");

  // NvBufferTransformParams, FILTER & CROP
  memset(&depth_right_trans_, 0, sizeof(NvBufferTransformParams));
  depth_right_trans_.transform_flag = NVBUFFER_TRANSFORM_CROP_SRC | NVBUFFER_TRANSFORM_FILTER;
  depth_right_trans_.transform_filter = NvBufferTransform_Filter_Nicest;
  depth_right_trans_.src_rect.top = 0;
  depth_right_trans_.src_rect.left = camera_width;
  depth_right_trans_.src_rect.width = camera_width;
  depth_right_trans_.src_rect.height = camera_height;
  memset(&depth_left_trans_, 0, sizeof(NvBufferTransformParams));
  depth_left_trans_.transform_flag = NVBUFFER_TRANSFORM_CROP_SRC | NVBUFFER_TRANSFORM_FILTER;
  depth_left_trans_.transform_filter = NvBufferTransform_Filter_Nicest;
  depth_left_trans_.src_rect.top = 0;
  depth_left_trans_.src_rect.left = 0;
  depth_left_trans_.src_rect.width = camera_width;
  depth_left_trans_.src_rect.height = camera_height;

  // queue buff
  struct v4l2_buffer buf;
  memset(&buf, 0, sizeof(buf));
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  buf.index = 0;
  if (ioctl(cam_fd_, VIDIOC_QUERYBUF, &buf) < 0)
    SPDLOG_CRITICAL("VIDIOC_QUERYBUF");
  capture_jpeg_buffer_.size = buf.length;
  capture_jpeg_buffer_.start = (unsigned char*)mmap(NULL /* start anywhere */,
                                                    buf.length, PROT_READ | PROT_WRITE /* required */,
                                                    MAP_SHARED /* recommended */,
                                                    cam_fd_, buf.m.offset);
  if (MAP_FAILED == capture_jpeg_buffer_.start)
    SPDLOG_CRITICAL("Failed to map buffers");
  if (0 != ioctl(cam_fd_, VIDIOC_QBUF, &buf))
    SPDLOG_CRITICAL("VIDIOC_QBUF");
  // start_capturing
  enum v4l2_buf_type type;
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (0 != ioctl(cam_fd_, VIDIOC_STREAMON, &type))
    SPDLOG_CRITICAL("VIDIOC_STREAMON");
  // warm up
  usleep(3000000);  // wait for hardware?
  Capture();        // skip the first frame
  Capture();
  Encode();
  Detect();
  Depth();
  // test threads
  RunParallel();
  return 0;
}

int Camera::RunParallel() {
  Capture();
  std::thread encode_thread(&Camera::Encode, this);
  Detect();
  Depth();
  encode_thread.join();
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
  if (0 != ioctl(cam_fd_, VIDIOC_DQBUF, &v4l2_buf))
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
  if (0 != ioctl(cam_fd_, VIDIOC_QBUF, &v4l2_buf))
    SPDLOG_CRITICAL("VIDIOC_QBUF");
  SPDLOG_TRACE("End");
  return 0;
}

int Camera::Encode() {
  SPDLOG_TRACE("Strat");
  SPDLOG_TRACE("NvBufferTransform");  // YUV422 & stereo -> YUV420 & right
  if (0 != NvBufferTransform(capture_yuv_fd_, encode_yuv_fd_, &depth_left_trans_))
    SPDLOG_CRITICAL("Failed to convert the buffer");
  SPDLOG_TRACE("Encode");
  encode_index = 1 - encode_index;
  encode_jpeg_list[encode_index].size = encode_jpeg_max_size_;
  if (0 != jpegenc_->encodeFromFd(encode_yuv_fd_, JCS_YCbCr,
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
  SPDLOG_TRACE("NvBufferTransform");  // YUV422 & stereo -> RGBA & right, left
  if (0 != NvBufferTransform(capture_yuv_fd_, depth_left_fd_, &depth_left_trans_))
    SPDLOG_CRITICAL("Failed to convert the buffer");
  if (0 != NvBufferTransform(capture_yuv_fd_, depth_right_fd_, &depth_right_trans_))
    SPDLOG_CRITICAL("Failed to convert the buffer");
  // cudaDeviceSynchronize();

  SPDLOG_TRACE("Convert");  // RGBA -> NV12_ER
  if (0 != vpiSubmitConvertImageFormat(depth_stream_, VPI_BACKEND_CUDA, depth_left_img_, depth_left_ER_img_, &depth_convParams_))
    SPDLOG_CRITICAL("vpiSubmitConvertImageFormat");
  if (0 != vpiSubmitConvertImageFormat(depth_stream_, VPI_BACKEND_CUDA, depth_right_img_, depth_right_ER_img_, &depth_convParams_))
    SPDLOG_CRITICAL("vpiSubmitConvertImageFormat");

  SPDLOG_TRACE("DeNoise");
  if (0 != vpiSubmitTemporalNoiseReduction(depth_stream_, VPI_BACKEND_CUDA, depth_tnr_, NULL, depth_left_ER_img_, depth_left_nr_img_, &depth_tnr_params_))
    SPDLOG_CRITICAL("vpiSubmitTemporalNoiseReduction");
  if (0 != vpiSubmitTemporalNoiseReduction(depth_stream_, VPI_BACKEND_CUDA, depth_tnr_, depth_left_nr_img_, depth_right_ER_img_, depth_right_nr_img_, &depth_tnr_params_))
    SPDLOG_CRITICAL("vpiSubmitTemporalNoiseReduction");

  SPDLOG_TRACE("StereoDisparity");
  // if (0 != vpiSubmitStereoDisparityEstimator(depth_stream_, VPI_BACKEND_CUDA, depth_stereo_, depth_left_nr_img_, depth_right_nr_img_, depth_disparity_, depth_confidenceMap_, NULL))
  //   SPDLOG_CRITICAL("vpiSubmitStereoDisparityEstimator");
  if (0 != vpiSubmitStereoDisparityEstimator(depth_stream_, VPI_BACKEND_CUDA, depth_stereo_, depth_left_nr_img_, depth_right_nr_img_, depth_disparity_, NULL, NULL))
    SPDLOG_CRITICAL("vpiSubmitStereoDisparityEstimator");
  SPDLOG_TRACE("vpiStreamSync");
  if (0 != vpiStreamSync(depth_stream_))
    SPDLOG_CRITICAL("vpiStreamSync");

  SPDLOG_TRACE("weighted disparity");
  const int box_index = model_.buffer_index;
  model_.depth[box_index].clear();
  for (int i : model_.indices[box_index]) {
    // TODO: CUDA
    float depth = 0;
    const int x = model_.boxes[box_index][i].x / depth_factor_;
    const int y = model_.boxes[box_index][i].y / depth_factor_;
    const int width = model_.boxes[box_index][i].width / depth_factor_;
    const int height = model_.boxes[box_index][i].height / depth_factor_;
    uint16_t depth_list[width * height] = {0};
    uint64_t sum = 0;
    // uint64_t weight = 0;
    size_t list_i = 0;
    for (size_t y_i = y; y_i < y + height; y_i++) {
      for (size_t x_i = x; x_i < x + width; x_i++) {
        // weight += ((uint16_t*)depth_confidenceMap_data_)[y_i * (camera_width_ / depth_factor_) + x_i] / 256;
        // sum += (((uint16_t*)depth_confidenceMap_data_)[y_i * (camera_width_ / depth_factor_) + x_i]) * (((uint16_t*)depth_disparity_data_)[y_i * (camera_width_ / depth_factor_) + x_i] / 256);
        if (((uint16_t*)depth_disparity_data_)[y_i * (camera_width_ / depth_factor_) + x_i]) {
          depth_list[list_i++] = ((uint16_t*)depth_disparity_data_)[y_i * (camera_width_ / depth_factor_) + x_i];
        }
      }
    }
    std::sort(depth_list, depth_list + list_i,std::greater<uint16_t>());
    size_t l_i;
    for (l_i = 0; l_i < list_i * 0.1 + 1; l_i++) {
      sum += depth_list[l_i];
    }
    depth = depth_k_ / (sum / l_i) + depth_b_;
    model_.depth[box_index].emplace_back(depth);
  }

  // uint16_t tmp[2560/2/2][720/2];
  //   uint16_t tmp2[2560/2/2][720/2];
  // cudaMemcpy(tmp, depth_disparity_data_, 2560/2/2 * 720/2 * 2, cudaMemcpyDeviceToHost);
  // cudaMemcpy(tmp2, depth_confidenceMap_data_, 2560/2/2 * 720/2 * 2, cudaMemcpyDeviceToHost);
  // cv::Mat cvDisparityColor;
  // cv::Mat cvmat1(360, 640, CV_16UC1, depth_confidenceMap_data_);
  // cv::imwrite("confidence.bmp", cvmat1);
  // cv::Mat cvDisparity(360, 640, CV_16UC1, depth_disparity_data_);
  // cvDisparity.convertTo(cvDisparity, CV_8UC1, 255.0 / (32 * 64), 0);
  // applyColorMap(cvDisparity, cvDisparityColor, cv::COLORMAP_JET);
  // cv::imwrite("disparity.bmp", cvDisparityColor);

  SPDLOG_TRACE("End");
  return 0;
}

int Camera::Detect() {
  SPDLOG_TRACE("Strat");

  SPDLOG_TRACE("NvBufferTransform");  // YUV422 & stereo -> rgb & right
  if (0 != NvBufferTransform(capture_yuv_fd_, detect_rbg_fd_, &depth_left_trans_))
    SPDLOG_CRITICAL("Failed to convert the buffer");

  SPDLOG_TRACE("Read Fd");
  EGLImageKHR egl_image = NvEGLImageFromFd(NULL, detect_rbg_fd_);
  if (egl_image == NULL)
    SPDLOG_CRITICAL("NvEGLImageFromFd");
  cudaEglFrame eglFrame;
  cudaGraphicsResource* eglResource = NULL;
  // cudaFree(0);
  if (CUDA_FAILED(cudaGraphicsEGLRegisterImage(&eglResource, egl_image,
                                               CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)))
    SPDLOG_CRITICAL("cudaGraphicsEGLRegisterImage");
  if (CUDA_FAILED(cudaGraphicsResourceGetMappedEglFrame(&eglFrame, eglResource, 0, 0)))
    SPDLOG_CRITICAL("cuGraphicsResourceGetMappedEglFrame");

  if (eglFrame.frameType != cudaEglFrameTypePitch)
    SPDLOG_CRITICAL("{} != cudaEglFrameTypePitch", eglFrame.frameType);

  SPDLOG_TRACE("Model");
  model_.Process((void*)eglFrame.frame.pArray[0]);

  // SPDLOG_TRACE("Synchronize");
  // cudaDeviceSynchronize();

  // uchar4* pdata;
  // NvBufferMemMap(detect_rbg_fd_, 0, NvBufferMem_Read, (void**)&pdata);
  // NvBufferMemSyncForCpu(detect_rbg_fd_, 0, (void**)&pdata);
  // cv::Mat cvmat3(640, 640, CV_8UC4, pdata);
  // cv::Mat rgb;
  // cv::cvtColor(cvmat3, rgb, cv::COLOR_RGBA2BGR);
  // cv::imwrite("frame.bmp", rgb);
  // NvBufferMemUnMap(detect_rbg_fd_, 0, (void**)&pdata);

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
  if (0 != ioctl(cam_fd_, VIDIOC_STREAMOFF, &type))
    SPDLOG_CRITICAL("VIDIOC_STREAMOFF");
  // uninit_device
  munmap(capture_jpeg_buffer_.start, capture_jpeg_buffer_.size);
  // close_device
  if (0 != close(cam_fd_))
    SPDLOG_CRITICAL("close");
  // free memory
  NvBufferDestroy(detect_rbg_fd_);
  NvBufferDestroy(encode_yuv_fd_);
  NvBufferDestroy(depth_left_fd_);
  NvBufferDestroy(depth_right_fd_);
  for (size_t i = 0; i < mjpeg_num; i++) {
    free(encode_jpeg_list[i].start);
  }
  vpiStreamDestroy(depth_stream_);
  vpiImageDestroy(depth_left_img_);
  vpiImageDestroy(depth_right_img_);
  vpiImageDestroy(depth_left_nr_img_);
  vpiImageDestroy(depth_right_nr_img_);
  vpiImageDestroy(depth_left_ER_img_);
  vpiImageDestroy(depth_right_ER_img_);
  vpiImageDestroy(depth_confidenceMap_);
  vpiImageDestroy(depth_disparity_);
  vpiPayloadDestroy(depth_stereo_);
  vpiPayloadDestroy(depth_tnr_);
  cudaFree(depth_disparity_data_);
  cudaFree(depth_confidenceMap_data_);
  delete jpegdec_;
  delete jpegenc_;
}

}  // namespace jetson