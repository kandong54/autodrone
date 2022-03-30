#!/bin/bash

# prebuilt v1.17 brokes https://github.com/envoyproxy/envoy/issues/15235
VERSION="1.16"
DEB_NAME="getenvoy-envoy_$VERSION-1_arm64"
TMP="envoy_tmp"

rm -rf $TMP

mkdir $TMP
pushd $TMP
# extract arm64 version bin from docker
mkdir $TMP
pushd $TMP
docker pull --platform=linux/arm64 envoyproxy/envoy:v$VERSION-latest
docker save envoyproxy/envoy > envoy.tar
tar -xf envoy.tar
rm envoy.tar
for H in `find . -name "*.tar"`;  do
  tar -xf $H
done
mv usr/local/bin/envoy ../
popd
rm -rf $TMP
# unpack deb
apt download getenvoy-envoy
mkdir -p $DEB_NAME/DEBIAN
dpkg-deb -x getenvoy-envoy*.deb $DEB_NAME/
dpkg-deb -e getenvoy-envoy*.deb $DEB_NAME/DEBIAN
sed -i "s/Version: .*/Version: ${VERSION}/" $DEB_NAME/DEBIAN/control
sed -i 's/Architecture: amd64/Architecture: arm64/' $DEB_NAME/DEBIAN/control
# modify bin
mv envoy $DEB_NAME/usr/bin
# repack deb
dpkg-deb --build --root-owner-group $DEB_NAME
popd

# envoy -c tools/envoy.yaml