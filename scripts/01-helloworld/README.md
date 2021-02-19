Compile protobuffer

    python -m grpc_tools.protoc -I./ --python_out=. --grpc_python_out=. ./helloworld.proto

Compile server container

    cd cl-workshop-cvpr21-docker/scripts/01-helloworld/
    docker build -t grpctest-server -f docker-server/Dockerfile .

Compile client container

    # same cd
    docker build -t grpctest-client -f docker-client/Dockerfile . 

Run server interactively

    docker run -it --rm -p 50051:50051 grpctest-server

Create bridge network

    docker network create --driver bridge grpcnet

Run server connected to network (Has to be named and part of the same network. Name has to be entered on client side into address line)

    docker run -d --network grpcnet --name grpcserver  grpctest-server

Run client interactively against running server

     docker run -it --rm --network grpcnet grpctest-client

Run both containers with automatic networking:

    docker-compose up --build

