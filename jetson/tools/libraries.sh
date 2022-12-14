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

# yaml-cpp
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

# OpenCV
# https://github.com/dusty-nv/jetson-containers/blob/master/scripts/opencv_install_deps.shcat 
# optional
sudo apt-get update
sudo apt-get install -y --no-install-recommends build-essential gfortran cmake \
  git file tar libatlas-base-dev libavcodec-dev libavformat-dev libavresample-dev \
  libcanberra-gtk3-module libdc1394-22-dev libeigen3-dev libglew-dev \
  libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev libgstreamer1.0-dev \
  libgtk-3-dev libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev liblapack-dev liblapacke-dev \
  libopenblas-dev libpng-dev libpostproc-dev libswscale-dev libtbb-dev libtbb2 libtesseract-dev \
  libtiff-dev libv4l-dev libxine2-dev libxvidcore-dev libx264-dev libgtkglext1 libgtkglext1-dev \
  pkg-config qv4l2 v4l-utils zlib1g-dev
sudo apt-get install -y --no-install-recommends python3-pip python3-dev python3-numpy python3-distutils python3-setuptools
# https://github.com/dusty-nv/jetson-containers/blob/master/scripts/opencv_version.sh
# It takes hours to build opencv on jeston. So we'd better ues the prebuilt binaries
OPENCV_URL="https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz"
OPENCV_DEB="OpenCV-4.5.0-aarch64.tar.gz"
# https://github.com/dusty-nv/jetson-containers/blob/be0dca3d19b30e64d129c92e93425c9ede40d65f/scripts/opencv_install.sh
# remove previous OpenCV installation if it exists
apt-get purge -y '*opencv*' || echo "previous OpenCV installation not found"
# download and extract the deb packages
mkdir opencv
cd opencv
wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${OPENCV_URL} -O ${OPENCV_DEB}
tar -xzvf ${OPENCV_DEB}
# install the packages and their dependencies
dpkg -i --force-depends *.deb
apt-get update 
apt-get install -y -f --no-install-recommends
dpkg -i *.deb
rm -rf /var/lib/apt/lists/*
apt-get clean
# remove the original downloads
cd ../
rm -rf opencv
# manage some install paths
PYTHON3_VERSION=`python3 -c 'import sys; version=sys.version_info[:3]; print("{0}.{1}".format(*version))'`
local_include_path="/usr/local/include/opencv4"
local_python_path="/usr/local/lib/python${PYTHON3_VERSION}/dist-packages/cv2"
ln -s /usr/include/opencv4 $local_include_path
ln -s /usr/lib/python${PYTHON3_VERSION}/dist-packages/cv2 $local_python_path
# test importing cv2
echo "testing cv2 module under python..."
python3 -c "import cv2; print('OpenCV version:', str(cv2.__version__)); print(cv2.getBuildInformation())"

# jetson-inference
git clone --recursive https://github.com/dusty-nv/jetson-inference --depth 1
cd jetson-inference && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DENABLE_NVMM=ON
make -j$(nproc)
sudo make install
cd ../..
rm -rf jetson-utils

# jetson-utils
git clone https://github.com/dusty-nv/jetson-utils --depth 1
cd jetson-utils && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DENABLE_NVMM=ON
make -j$(nproc)
sudo make install
cd ../..
rm -rf jetson-utils

sudo ldconfig