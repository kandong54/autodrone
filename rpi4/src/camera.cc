#include "camera.h"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <ctime>
#include <chrono>
#include <thread>

Camera::Camera()
{
    cv::Mat frame;
    //--- INITIALIZE VIDEOCAPTURE
    cv::VideoCapture cap;
    // open the default camera using default API
    // cap.open(0);
    // OR advance usage: select any API backend
    int deviceID = 0;        // 0 = open default camera
    int apiID = cv::CAP_ANY; // 0 = autodetect default API
    // open selected camera using selected API
    cap.open(deviceID, apiID);
    // check if we succeeded
    if (!cap.isOpened())
    {
        std::cerr << "ERROR! Unable to open camera\n";
        //return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 960);
    // no buffer
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    //--- GRAB AND WRITE LOOP
    std::cout << "Start grabbing" << std::endl;
    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.grab(); // clear buffer TODO: any good solution?
        cap.read(frame);
        // check if we succeeded
        if (frame.empty())
        {
            std::cerr << "ERROR! blank frame grabbed" << std::endl;
            break;
        }
        // crop image
        cv::Rect myROI(160, 40, 800, 680); //960*720 -> 640*640
        cv::Mat croppedImage = frame(myROI);
        // save image
        char filename[100];
        std::time_t t_c = std::time(nullptr);
        std::strftime(filename, sizeof(filename), "%H%M%S.jpg", std::localtime(&t_c));
        if (!cv::imwrite(filename, croppedImage))
        {
            std::cerr << "ERROR! Unable to save image" << std::endl;
            break;
        }
        std::cout << filename << std::endl;
        // show live and wait for a key with timeout long enough to show images
        // imshow("Live", frame);
        // if (cv::waitKey(5) >= 0)
        //     break;
        //std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
}