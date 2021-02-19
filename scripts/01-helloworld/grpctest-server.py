import threading
from concurrent import futures
import logging

import grpc

import helloworld_pb2
import helloworld_pb2_grpc


class Greeter(helloworld_pb2_grpc.GreeterServicer):
    def __init__(self, done):
        super().__init__()
        self.done = done

    def SayHello(self, request, context):
        print("incoming sayhello req")
        return helloworld_pb2.HelloReply(message=f"Hello, {request.name}!")

    def SayHelloAgain(self, request, context):
        print("incoming sayhello again req")
        return helloworld_pb2.HelloReply(message=f"Sup again, {request.name}?")

    def Shutdown(self, request, context):
        print("shutting down server")
        self.done.set()
        return helloworld_pb2.Empty()


logging.basicConfig()
done = threading.Event()
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(done), server)
server.add_insecure_port("[::]:50051")
print("starting server")
server.start()
print("server running")
done.wait()  # wait until the done-ness triggers
# server.wait_for_termination()
