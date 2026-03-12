ARG VLLM_IMAGE=vllm/vllm-openai:v0.17.0
FROM ${VLLM_IMAGE}

# vllm/vllm-openai has ENTRYPOINT ["vllm","serve"], which would treat our CMD as model_tag.
# Clear entrypoint so we can run our own wrapper process.
ENTRYPOINT []

WORKDIR /app

# The generated *_pb2.py files are produced by an older protoc, so we pin protobuf for compatibility.
# vLLM 0.17.0 requires protobuf>=5.29.6, so keep this on a 5.x version rather than the older 3.x pin.
RUN python3 -m pip install --no-cache-dir \
    grpcio \
    httpx \
    protobuf==5.29.6

COPY . /app

ENV PYTHONUNBUFFERED=1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# gRPC + vLLM(OpenAI) ports
EXPOSE 50051 30000

CMD ["python3", "-u", "/app/vllm_endpointing_grpc_service.py"]
