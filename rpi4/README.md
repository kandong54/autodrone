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

To cross complie TFLite, run [rpi4/tools/build_tflite.sh](tools/build_tflite.sh) on the host computer.
To install its dependencies:
```shell
sudo apt install libabsl-dev libflatbuffers-dev
```


## Reference
- [Build TensorFlow Lite for ARM boards](https://www.tensorflow.org/lite/guide/build_arm)
- [PKGBUILD libtensorflow-lite - AUR](https://aur.archlinux.org/cgit/aur.git/tree/PKGBUILD?h=libtensorflow-lite)
- [Install Precompiled TensorFlow Lite 2.8 on Raspberry Pi](https://lindevs.com/install-precompiled-tensorflow-lite-on-raspberry-pi/)
- [Building binary deb packages: a practical guide](https://www.internalpointers.com/post/build-binary-deb-package-practical-guide)