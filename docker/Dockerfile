# Build command verifies this local base's exact image ID against pins.json.
ARG BASE=qwen38-int8-lab/eval:0.1.0
FROM ${BASE}

# FP8 KV on SM86 uses FlashInfer's runtime JIT path.  The base evaluation
# image has gcc but intentionally omits the C++ frontend required by nvcc.
# Keep this small, pinned compiler addition in the evaluation overlay so the
# same image can run both native lm-eval and FP8 profile stages.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      g++=4:13.2.0-7ubuntu1 \
      g++-13=13.3.0-6ubuntu2~24.04.1 \
      g++-13-x86-64-linux-gnu=13.3.0-6ubuntu2~24.04.1 \
      g++-x86-64-linux-gnu=4:13.2.0-7ubuntu1 \
      libstdc++-13-dev=13.3.0-6ubuntu2~24.04.1 && \
    rm -rf /var/lib/apt/lists/*

COPY docker/evalplus-requirements.lock /opt/evalplus-requirements.lock
RUN uv venv --python python --no-python-downloads /opt/evalplus && uv pip install --python /opt/evalplus/bin/python --no-cache \
    -r /opt/evalplus-requirements.lock \
    && uv pip check --python /opt/evalplus/bin/python
WORKDIR /app
ENTRYPOINT ["python"]
