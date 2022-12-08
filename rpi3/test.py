#pip3 install grpcio-tools
#python3 -m grpc_tools.protoc --python_out=. --pyi_out=. --grpc_python_out=. --proto_path=./protos drone.proto
import grpc
import time
import drone_pb2
import drone_pb2_grpc

# init connection
channel = grpc.insecure_channel('127.0.0.1:9090')
stub = drone_pb2_grpc.DroneStub(channel)
# get image size
image_size = stub.GetImageSize(drone_pb2.Empty())
print(image_size)
for _ in range(10):
    time.sleep(1)
    # get bounding boxes
    box = stub.GetBox(drone_pb2.Empty())
    print(box)