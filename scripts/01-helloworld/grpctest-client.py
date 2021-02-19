import logging

import grpc

import helloworld_pb2
import helloworld_pb2_grpc

logging.basicConfig()

# with grpc.insecure_channel("localhost:50051") as channel:
with grpc.insecure_channel("grpcserver:50051") as channel:
    stub = helloworld_pb2_grpc.GreeterStub(channel)
    response = stub.SayHello(helloworld_pb2.HelloRequest(name="you"))
    print("Greeter client received: " + response.message)
    response = stub.SayHelloAgain(helloworld_pb2.HelloRequest(name="fabster"))
    print("Greeter client received: " + response.message)

    stub.Shutdown(helloworld_pb2.Empty())
