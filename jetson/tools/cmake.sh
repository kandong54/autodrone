#!/bin/bash
# https://forums.developer.nvidia.com/t/having-problems-updating-cmake-on-xavier-nx/169265/2


# https://github.com/Kitware/CMake/releases/
VERSION="3.23.4"
TMP="cmake_tmp"

rm -rf $TMP

mkdir $TMP
pushd $TMP
# download pre-built binaries
wget https://github.com/Kitware/CMake/releases/download/v$VERSION/cmake-$VERSION-Linux-aarch64.tar.gz -q --show-progress 
# unzip
tar -zxvf cmake-$VERSION-Linux-aarch64.tar.gz 
pushd cmake-$VERSION-linux-aarch64
# copy files
sudo cp -rf bin/ doc/ share/ /usr/local/
sudo cp -rf man/* /usr/local/man
sync
popd
popd
rm -rf $TMP
# check version
cmake --version 