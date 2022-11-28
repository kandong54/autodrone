# (Deprecated) Drone App
The drone app is a C++ program on Raspberry Pi 4 Model B.  It is used to control the drone and serve data to the UI.

Up to now, it can 
- load the config
- capture camera images
- detect flowers in an image
- serve data to the UI with auth

## Dependencies

- OpenCV
- gRPC
- TFLite
- spdlog
- OpenSSL
- yaml-cpp

To install these:

```shell
sudo apt install libopencv-dev libgrpc++-dev libprotobuf-dev libspdlog-dev libssl-dev libabsl-dev libflatbuffers-dev libyaml-cpp-dev
sudo apt install protobuf-compiler-grpc 
```

To cross complie TFLite, run [build_tflite.sh](tools/build_tflite.sh) on the host computer.

## Build and Debug
See [Cross Compile and Remote Debug](docs/cross_compile_and_remote_debug.md) for detailed instructions.

[Config](tools/config.yaml) is required to run the app. You may also need to create [certificates](https://grpc.github.io/grpc/cpp/structgrpc_1_1_ssl_server_credentials_options.html) and [model](/model/).

## Server

Due to browser limitations, [gRPC-web](https://github.com/grpc/grpc-web) needs a special proxy to connect to gRPC services; by default, gRPC-web uses [Envoy](https://www.envoyproxy.io/). To run Envoy with the [config](tools/envoy.yaml):

```shell
envoy -c envoy.yaml
```

Because of the firewall, Envoy is now hosted on a public server instead of the Raspberry Pi. Server and Raspberry Pi transfer data via [WireGuard](https://www.wireguard.com/). And the data transfers between users and server is encrypted by SSL/TLS. 

```
RPi <--- WireGuard ---> Server(Envoy) <--- SSL/TLS ---> Users
```

Envoy will be hosted on the Raspberry Pi if it has a public IP without firewall. The script [extract_envoy.sh](tools/extract_envoy.sh) can extract the prebuilt arm64 Envoy from Docker images. And Envoy >= v1.17 doesn't work on RPi because of TCMalloc: envoyproxy/envoy#15235.

## GPU
Now the image compression and model inference are run on the CPU. However, GPU can be used for acceleration. The branch [h264](https://github.com/kandong54/autodrone/tree/h264) shows how to use GPU to encode images.
