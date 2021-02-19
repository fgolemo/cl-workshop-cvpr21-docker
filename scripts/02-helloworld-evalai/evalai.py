import docker

client = docker.from_env()
server = client.containers.run("grpctest-server", detach=True, name="grpcserver", network="grpcnet")
results = client.containers.run("grpctest-client", name="grpcclient", network="grpcnet")
print(results)
# HERE WE REPORT THE RESULTS TO EVALAI

print("done, killing containers")
server.stop()
