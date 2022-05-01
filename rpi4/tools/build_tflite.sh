#!/bin/bash

# Reference
# https://www.tensorflow.org/lite/guide/build_arm
# https://aur.archlinux.org/cgit/aur.git/tree/PKGBUILD?h=libtensorflow-lite
# https://lindevs.com/install-precompiled-tensorflow-lite-on-raspberry-pi/
# https://www.internalpointers.com/post/build-binary-deb-package-practical-guide

VERSION="2.8.0"
SRC_PATH="tensorflow_src"
DEB_NAME="libtensorflowlite-dev_$VERSION-1_arm64"
BUILD_PATH="$(pwd)/$DEB_NAME"

# Remove existed files
rm -rf $BUILD_PATH $SRC_PATH $DEB_NAME.deb

# Clone TensorFlow repository
git clone -b "v$VERSION" --depth 1 --shallow-submodules https://github.com/tensorflow/tensorflow "$SRC_PATH"
pushd $SRC_PATH

# Build ARM binary
# bazel clean
#  -mfp16-format=ieee 
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so # C++ library
# bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so # C library

# Copy Libraries
LIB_PATH="$BUILD_PATH/usr/lib/aarch64-linux-gnu"
mkdir -p $LIB_PATH
cp bazel-bin/tensorflow/lite/libtensorflowlite.so $LIB_PATH
# cp bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so $LIB_PATH

# Extract All Header Files
INC_PATH="$BUILD_PATH/usr/include"
mkdir -p $INC_PATH
# TensorFlow Lite
cp --parents tensorflow/core/public/version.h $INC_PATH
LITE_PATH="${INC_PATH}/tensorflow/lite"
mkdir -p $LITE_PATH
pushd tensorflow/lite/
for H in `find . -name "*.h"`;  do
  PathFileDir=$(dirname $H)
  mkdir -p "${LITE_PATH}/${PathFileDir}"     # no error, make parents too
  cp "$H" "${LITE_PATH}/${PathFileDir}/"    # preserve ownership...
done
popd
popd

# Package
mkdir -p $LIB_PATH/pkgconfig
cat > $LIB_PATH/pkgconfig/tflite.pc <<EOF
prefix=/usr
includedir=\${prefix}/include
libdir=\${prefix}/lib/aarch64-linux-gnu

Name: TFLite
Description: Tensorflow Lite
Version: $VERSION
Cflags: -I\${includedir} 
Libs: -L\${libdir} -ltensorflowlite
EOF
mkdir -p $BUILD_PATH/DEBIAN
cat > $BUILD_PATH/DEBIAN/control <<EOF
Package: libtensorflowlite-dev
Version: $VERSION
Section: devel 
Priority: optional 
Architecture: arm64
Maintainer: TensorFlow
Description: Tensorflow Lite - Library and Headers.
EOF

dpkg-deb --build --root-owner-group $DEB_NAME
