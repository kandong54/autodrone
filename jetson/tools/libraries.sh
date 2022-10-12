#!/bin/bash
# https://forums.developer.nvidia.com/t/having-problems-updating-cmake-on-xavier-nx/169265/2

sudo apt update
sudo apt install libssl-dev

# spdlog
git clone https://github.com/gabime/spdlog --depth 1 --branch v1.10.0
cd spdlog && mkdir build && cd build
cmake .. -DSPDLOG_BUILD_SHARED=ON \
         -DSPDLOG_BUILD_EXAMPLE=OFF
make -j$(nproc)
sudo make install
cd ../..
rm -rf spdlog

# spdlog
git clone https://github.com/jbeder/yaml-cpp --depth 1 --branch yaml-cpp-0.7.0
cd yaml-cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DYAML_BUILD_SHARED_LIBS=ON \
         -DYAML_CPP_INSTALL=ON \
         -DYAML_CPP_BUILD_TESTS=OFF
make -j$(nproc)
sudo make install
cd ../..
rm -rf yaml-cpp

# gRPC
git clone https://github.com/grpc/grpc --depth 1 --branch v1.49.1
cd grpc
git submodule update --init --depth 1
mkdir -p cmake/build
pushd cmake/build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DgRPC_INSTALL=ON \
  -DgRPC_BUILD_TESTS=OFF \
  -DgRPC_BUILD_CSHARP_EXT=OFF \
  -DgRPC_SSL_PROVIDER=package \
  ../..
LD_LIBRARY_PATH_BAK=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
make -j$(nproc)
sudo make install
popd
cd ..
rm -rf grpc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_BAK

sudo ldconfig