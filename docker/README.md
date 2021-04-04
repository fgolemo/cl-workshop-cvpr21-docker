Build base container

    cd docker/base
    docker build -t cl21base .

Tag & push container

    docker tag cl21base fgolemo/cl21base:v1
    docker push fgolemo/cl21base:v1
