#!/bin/bash

# getenvoy-deb: https://cloudsmith.io/~tetrate/repos/getenvoy-deb-stable/setup/#formats-deb
# curl -1sLf 'https://deb.dl.getenvoy.io/public/setup.deb.sh' | sudo -E bash

# prebuilt v1.17 brokes for aarch64 https://github.com/envoyproxy/envoy/issues/15235
VERSION="1.16"
DEB_NAME="getenvoy-envoy_$VERSION-1_arm64"
TMP="envoy_tmp"

rm -rf $TMP

mkdir $TMP
pushd $TMP
# extract arm64 version bin from docker
mkdir $TMP
pushd $TMP
# download the docker
docker pull --platform=linux/arm64 envoyproxy/envoy:v$VERSION-latest
# get the binary file
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
#apt update
#apt download getenvoy-envoy
# https://cloudsmith.io/~tetrate/repos/getenvoy-deb-stable/packages/
wget https://deb.dl.getenvoy.io/public/deb/any-distro/pool/any-version/main/g/ge/getenvoy-envoy_1.16.2.p0.ge98e41a-1p71.gbe6132a_amd64.deb
mkdir -p $DEB_NAME/DEBIAN
dpkg-deb -x getenvoy-envoy*.deb $DEB_NAME/
dpkg-deb -e getenvoy-envoy*.deb $DEB_NAME/DEBIAN
sed -i "s/Version: .*/Version: ${VERSION}/" $DEB_NAME/DEBIAN/control
sed -i 's/Architecture: amd64/Architecture: arm64/' $DEB_NAME/DEBIAN/control
# modify bin
mv envoy $DEB_NAME/usr/bin
# repack deb
dpkg-deb --build --root-owner-group $DEB_NAME
# install
apt install ./$DEB_NAME.deb -y
popd
rm -rf $TMP

# envoy -c tools/envoy.yaml