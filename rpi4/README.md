# Raspberry Pi 4 Model B
Hello, World!

## Dependencies
- OpenCV
- gRPC
- TFLite

To install OpenCV and gRPC:
```shell
sudo apt install libopencv-dev libgrpc++-dev protobuf-compiler-grpc
```

To cross complie TFLite, run [build_tflite.sh](tools/build_tflite.sh) on the host computer.
To install its dependencies:
```shell
sudo apt install libabsl-dev libflatbuffers-dev
```

## Server

Due to browser limitations, gRPC-web needs a special proxy to connect to gRPC services; by default, gRPC-web uses [Envoy](https://www.envoyproxy.io/).

To extract prebuilt Envoy from Docker images, run [extract_envoy.sh](tools/extract_envoy.sh). To run Envoy with the [config](tools/envoy.yaml):

```shell
envoy -c envoy.yaml
```

Currently, Envoy is hosted on the Raspberry Pi and will be hosted on a server if possible.

The prebuilt Envoy >= v1.17 doesn't work on RPi because of TCMalloc: envoyproxy/envoy#15235.

## Reference
- [Build TensorFlow Lite for ARM boards](https://www.tensorflow.org/lite/guide/build_arm)
- [PKGBUILD libtensorflow-lite - AUR](https://aur.archlinux.org/cgit/aur.git/tree/PKGBUILD?h=libtensorflow-lite)
- [Install Precompiled TensorFlow Lite 2.8 on Raspberry Pi](https://lindevs.com/install-precompiled-tensorflow-lite-on-raspberry-pi/)
- [Building binary deb packages: a practical guide](https://www.internalpointers.com/post/build-binary-deb-package-practical-guide)
- [gRPC for Web Clients](https://github.com/grpc/grpc-web)