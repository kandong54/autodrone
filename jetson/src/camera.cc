#include "camera.h"

#include <spdlog/spdlog.h>

namespace jetson {

Camera::Camera(YAML::Node& config) : config_(config) {

}

int Camera::Open() {
  return 0;
}

int Camera::Capture() {
  return 0;
}

int Camera::Encode() {
  return 0;
}

bool Camera::IsOpened() {
}

Camera::~Camera() {
}

}  // namespace jetson