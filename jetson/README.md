# Drone App
The drone app is a C++ program on [Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano-developer-kit). It is used to control the drone and serve data to the UI.

Up to now, it can 
- load the config
- capture camera images
- detect flowers in an image
- serve data to the UI with auth

## Dependencies

- OpenCV
- gRPC
- TensorRT
- spdlog
- OpenSSL
- yaml-cpp
- jetson-utils

Most of them are imported by FetchContent of CMake.

To install OpenSSL library:

```shell
sudo apt install libssl-dev
```
To install jetson-inference and jetson-utils: [jetson-inference](
https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md)

## Run

[Config](tools/config.yaml) is required to run the app. You may also need to create [certificates](/tools/cert/) and [model](/model/).

## Server

Due to browser limitations, [gRPC-web](https://github.com/grpc/grpc-web) needs a special proxy to connect to gRPC services; by default, gRPC-web uses [Envoy](https://www.envoyproxy.io/). To run Envoy with the [config](tools/envoy.yaml):

```shell
envoy -c envoy.yaml
```

## Setup Jetson
Booting Jetson Nano from a flash device makes everything smooth: https://github.com/jetsonhacksnano/bootFromUSB