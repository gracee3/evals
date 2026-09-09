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

# Keep IFBench's verifier dependencies out of the pinned lm-eval/vLLM site.
COPY docker/ifbench-requirements.lock /opt/ifbench-requirements.lock
RUN uv venv --python python --no-python-downloads /opt/ifbench && uv pip install --python /opt/ifbench/bin/python --no-cache \
    -r /opt/ifbench-requirements.lock \
    && uv pip check --python /opt/ifbench/bin/python

# LiveCodeBench's official checker is kept in a separate environment. The
# wrapper exposes the base image's torch/vLLM site explicitly at invocation and
# adds only evaluator dependencies; generated code still runs inside the
# supervisor's no-network grading container.
COPY docker/livecodebench-requirements.lock /opt/livecodebench-requirements.lock
RUN uv venv --python python --no-python-downloads /opt/livecodebench && uv pip install --python /opt/livecodebench/bin/python --no-cache \
    -r /opt/livecodebench-requirements.lock \
    && uv pip install --python /opt/livecodebench/bin/python --no-cache --no-deps \
    'https://github.com/LiveCodeBench/LiveCodeBench/archive/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24.tar.gz'
# The upstream wheel's setuptools list omits evaluator subpackages; retain the
# pinned source tree so lcb_runner.evaluation is available to the wrapper.
ADD https://github.com/LiveCodeBench/LiveCodeBench/archive/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24.tar.gz /tmp/livecodebench.tar.gz
RUN mkdir -p /opt/livecodebench-src && tar -xzf /tmp/livecodebench.tar.gz --strip-components=1 -C /opt/livecodebench-src && rm -f /tmp/livecodebench.tar.gz
WORKDIR /app
ENTRYPOINT ["python"]
