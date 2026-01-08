ARG SGLANG_IMAGE=lmsysorg/sglang:latest
FROM ${SGLANG_IMAGE}

WORKDIR /app

# The generated *_pb2.py files are produced by an older protoc, so we pin protobuf for compatibility.
RUN python3 -m pip install --no-cache-dir \
    grpcio \
    httpx \
    protobuf==3.20.3

COPY . /app

ENV PYTHONUNBUFFERED=1

EXPOSE 50051 30000

CMD ["python3", "-u", "/app/sglang_endpointing_grpc_service.py"]

