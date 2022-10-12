# Drone App
The drone app is a C++ program on [Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano-developer-kit). It is used to control the drone and serve data to the UI.

Up to now, it can 
- load the config
- capture camera images
- detect flowers in an image
- serve data to the UI with auth

## Build
### Dependencies

- gRPC
- TensorRT
- spdlog
- OpenSSL
- yaml-cpp
- jetson-utils

Run this [script](tools/libraries.sh) to build and install those libraries.

To install jetson-inference and jetson-utils: [jetson-inference](
https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md)

### Cmake
Run this [script](tools/cmake.sh) to install the newer version of Cmake. [Ref](https://forums.developer.nvidia.com/t/having-problems-updating-cmake-on-xavier-nx/169265/2)
## Run

[Config](tools/config.yaml) is required to run the app. You may also need to create [certificates](/tools/cert/) and [model](/model/).

## Server

Due to browser limitations, [gRPC-web](https://github.com/grpc/grpc-web) needs a special proxy to connect to gRPC services; by default, gRPC-web uses [Envoy](https://www.envoyproxy.io/). Run this [script](tools/envoy.sh) to extract and install Envoy. To run Envoy with the [config](tools/envoy.yaml):

```shell
envoy -c envoy.yaml
```

## Setup Jetson
Booting Jetson Nano from a flash device makes everything smooth: https://github.com/jetsonhacksnano/bootFromUSB