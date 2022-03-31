#include "camera.h"

#include <iostream>
#include <ctime>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

namespace rpi4
{

  Camera::Camera()
  {
    int deviceID = 0;        // 0 = open default camera
    int apiID = cv::CAP_ANY; // 0 = autodetect default API
    cap_.open(deviceID, apiID);
    // check if we succeeded
    if (!cap_.isOpened())
    {
      std::cerr << "ERROR! Unable to open camera\n";
      //return -1;
    }
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, 960);
    // reduce buffer size so that we can read the most latest one
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
  }

  bool Camera::CaptureImage()
  {
    cv::Mat frame;

    // wait for a new frame from camera and store it into 'frame'
    // clear buffer
    // TODO: any good solution? How to read the most latest one?
    cap_.grab();
    cap_.read(frame);
    // check if we succeeded
    if (frame.empty())
    {
      std::cerr << "ERROR! blank frame grabbed" << std::endl;
      return false;
    }
    // TODO: add resize method.
    // crop image: 960*720 -> 640*640
    cv::Rect myROI(160, 40, 800, 680);
    cv::Mat croppedImage = frame(myROI);
    // save image
    char filename[100];
    std::time_t t_c = std::time(nullptr);
    std::strftime(filename, sizeof(filename), "%H%M%S.jpg", std::localtime(&t_c));
    if (!cv::imwrite(filename, croppedImage))
    {
      std::cerr << "ERROR! Unable to save image" << std::endl;
      return false;
    }
    std::cout << filename << std::endl;
    return true;
  }
} // namespace rpi4