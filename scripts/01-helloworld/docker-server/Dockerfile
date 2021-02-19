FROM python:3.8-slim
EXPOSE 50051
RUN pip install grpcio grpcio-tools
WORKDIR /work
# next line is just a safety measure to
COPY grpctest-server.py .
COPY ./*.py ./
CMD ["python", "grpctest-server.py"]