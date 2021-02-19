FROM python:3.8-slim
RUN pip install grpcio grpcio-tools
WORKDIR /work
# next line is just a safety measure to assure the context is right
COPY grpctest-client.py .
COPY ./*.py ./
CMD ["python", "grpctest-client.py"]