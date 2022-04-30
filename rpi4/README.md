# Raspberry Pi 4 Model B

Hello, World!

## Dependencies

- OpenCV
- gRPC
- TFLite
- spdlog
- OpenSSL
- FFmpeg
- yaml-cpp

To install these:

```shell
sudo apt install libopencv-dev libgrpc++-dev libprotobuf-dev libspdlog-dev libssl-dev libabsl-dev libflatbuffers-dev ffmpeg libyaml-cpp-dev
sudo apt install protobuf-compiler-grpc 
```

To cross complie TFLite, run [build_tflite.sh](tools/build_tflite.sh) on the host computer.

## Server

Due to browser limitations, gRPC-web needs a special proxy to connect to gRPC services; by default, gRPC-web uses [Envoy](https://www.envoyproxy.io/). To run Envoy with the [config](tools/envoy.yaml):

```shell
envoy -c envoy.yaml
```

Because of the firewall, Envoy is now hosted on a public server instead of the Raspberry Pi. Server and Raspberry Pi transfer data via [WireGuard](https://www.wireguard.com/). And the data transfers between users and server is encrypted by SSL/TLS. 

```
RPi <--- WireGuard ---> Server(Envoy) <--- SSL/TLS ---> Users
```

Envoy will be hosted on the Raspberry Pi if it has a public IP without firewall. The script [extract_envoy.sh](tools/extract_envoy.sh) can extract the prebuilt arm64 Envoy from Docker images. And Envoy >= v1.17 doesn't work on RPi because of TCMalloc: envoyproxy/envoy#15235.

## Reference
- [Build TensorFlow Lite for ARM boards](https://www.tensorflow.org/lite/guide/build_arm)
- [PKGBUILD libtensorflow-lite - AUR](https://aur.archlinux.org/cgit/aur.git/tree/PKGBUILD?h=libtensorflow-lite)
- [Install Precompiled TensorFlow Lite 2.8 on Raspberry Pi](https://lindevs.com/install-precompiled-tensorflow-lite-on-raspberry-pi/)
- [Building binary deb packages: a practical guide](https://www.internalpointers.com/post/build-binary-deb-package-practical-guide)
- [gRPC for Web Clients](https://github.com/grpc/grpc-web)