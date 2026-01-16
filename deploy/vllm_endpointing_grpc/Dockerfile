ARG VLLM_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_IMAGE}

# vllm/vllm-openai has ENTRYPOINT ["vllm","serve"], which would treat our CMD as model_tag.
# Clear entrypoint so we can run our own wrapper process.
ENTRYPOINT []

WORKDIR /app

# The generated *_pb2.py files are produced by an older protoc, so we pin protobuf for compatibility.
RUN python3 -m pip install --no-cache-dir \
    grpcio \
    httpx \
    protobuf==3.20.3

COPY . /app

ENV PYTHONUNBUFFERED=1

# gRPC + vLLM(OpenAI) ports
EXPOSE 50051 30000

CMD ["python3", "-u", "/app/vllm_endpointing_grpc_service.py"]

