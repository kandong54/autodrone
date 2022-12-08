# How to interact with Drone App in Rpi

## Disable Auth
To simply the interaction, we need to disable the auth in drone app. In [server.cc](../jetson/src/server.cc), replace server_creds by grpc::InsecureServerCredentials() in the AddListeningPort.

## Generate Files
Install the grpcio-tools library and generate the necessary files.
```
pip3 install grpcio-tools
python3 -m grpc_tools.protoc --python_out=. --pyi_out=. --grpc_python_out=. --proto_path=./protos drone.proto
```

## Code
The following codes show how to init the connection. You need to change the IP and port when necessay.
```
channel = grpc.insecure_channel('127.0.0.1:9090')
stub = drone_pb2_grpc.DroneStub(channel)
```
Now, we can get the image size:
```
image_size = stub.GetImageSize(drone_pb2.Empty())
```
We can also get the bounding boxes. The definition can be found in [proto](./protos/drone.proto).
```
box = stub.GetBox(drone_pb2.Empty())
```

# Example
[test](test.py)