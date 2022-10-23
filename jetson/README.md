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

### Model
Install necessary packages, PyTorch and Torchvision: [NVIDIA Jetson Nano Deployment](https://github.com/ultralytics/yolov5/issues/9627)

Install onnx
```shell
pip3 install onnx
```

Fix bugs in protobuf: copy this [file](https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py) to ~/.local/lib/python3.6/site-packages/google/protobuf/internal/builder.py [Ref](https://stackoverflow.com/a/74089097)

Convert the PyTorch model to ONNX
```shell
python3 export.py --weights yolov5n_flower.pt --include onnx
```

Convert the ONNX to TensorRT with fp16 
```shell
/usr/src/tensorrt/bin/trtexec --onnx=/home/jetson/yolov5/yolov5n_flower.onnx --saveEngine=yolov5n_flower.engine —fp16 --workspace=4096
```

## Run
[Config](tools/config.yaml) is required to run the app. You may also need to create [certificates](/tools/cert/) and [model](/model/).

## Server

Due to browser limitations, [gRPC-web](https://github.com/grpc/grpc-web) needs a special proxy to connect to gRPC services; by default, gRPC-web uses [Envoy](https://www.envoyproxy.io/). Run this [script](tools/envoy.sh) to extract and install Envoy. To run Envoy with the [config](tools/envoy.yaml):

```shell
envoy -c envoy.yaml
```

## Setup Jetson
Booting Jetson Nano from a flash device makes everything smooth: https://github.com/jetsonhacksnano/bootFromUSB