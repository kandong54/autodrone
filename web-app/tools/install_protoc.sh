#!/bin/bash

# Version
PB_VERSION="3.20.2"
WEB_VERSION="1.3.1"

# Clean
# rm $HOME/.local/bin/protoc*
# rm -r $HOME/.local/include/google/protobuf

# Protocol Buffers
PB_REL="https://github.com/protocolbuffers/protobuf/releases"
PB_FILE=protoc-$PB_VERSION-linux-x86_64.zip
curl -LO $PB_REL/download/v$PB_VERSION/$PB_FILE
unzip -q $PB_FILE -d $HOME/.local
rm $PB_FILE

# grpc-web
WEB_REL="https://github.com/grpc/grpc-web/releases"
WEB_FILE=protoc-gen-grpc-web-$WEB_VERSION-linux-x86_64
curl -LO $WEB_REL/download/$WEB_VERSION/$WEB_FILE
mv $WEB_FILE $HOME/.local/bin/protoc-gen-grpc-web
chmod +x $HOME/.local/bin/protoc-gen-grpc-web

# Update your environment’s path variable!
# export PATH="$PATH:$HOME/.local/bin"
