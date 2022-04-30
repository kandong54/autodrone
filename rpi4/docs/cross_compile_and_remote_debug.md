# Cross Compile and Remote Debug

System: Debian for WSL

## Create Arm64 Root Filesystem

Host:

```shell
sudo apt install sbuild ubuntu-dev-tools
mk-sbuild --arch=arm64 bullseye --name=rpi-bullseye  --distro=debian
```

Leave the ~/.sbuildrc default and restart session.

```shell
mk-sbuild --arch=arm64 bullseye --name=rpi-bullseye  --distro=debian
```

## Install Toolchain

Host:
```shell
sudo apt install build-essential g++-aarch64-linux-gnu
```

## Install Dependencies

### Host:

```shell
sudo apt install protobuf-compiler-grpc pkg-config
sudo sbuild-apt rpi-bullseye-arm64 apt-get install # lib
```

To install local deb, move files to /var/lib/schroot/chroots/rpi-bullseye-arm64/root

```shell
sudo sbuild-apt rpi-bullseye-arm64 apt-get install /root/*.deb
```

To fix broken symbolic links:
```shell
export LIB_PATH=/var/lib/schroot/chroots/rpi-bullseye-arm64/usr/lib/aarch64-linux-gnu
sudo rm $LIB_PATH/libblas.so.3
sudo ln -s $LIB_PATH/blas/libblas.so.3 $LIB_PATH/libblas.so.3 
sudo rm $LIB_PATH/liblapack.so.3
sudo ln -s $LIB_PATH/lapack/liblapack.so.3 $LIB_PATH/liblapack.so.3
```

To fix broken cmake links:
```shell
sudo nano /var/lib/schroot/chroots/rpi-bullseye-arm64/usr/lib/aarch64-linux-gnu/cmake/yaml-cpp/yaml-cpp-targets-none.cmake
```
Insert "/var/lib/schroot/chroots/rpi-bullseye-arm64" before "/usr"

### Raspberry Pi:

```shell
sudo apt install # lib
```

## Compile

Use [toolchain.cmake](../toolchain.cmake).

## Remote Debug

Raspberry Pi:

```shell
sudo apt install gdbserver
```

Host:

```shell
sudo apt install gdb-multiarch
```

Add  these two lines to ~/.gdbinit

```
set sysroot /var/lib/schroot/chroots/rpi-bullseye-arm64
set debug-file-directory /var/lib/schroot/chroots/rpi-bullseye-arm64/usr/lib/debug
```

## Reference
- [Ubuntu to Raspberry Pi OS Cross C++ Development](https://tttapa.github.io/Pages/Raspberry-Pi/C++-Development-RPiOS/index.html)