# Build command verifies this local base's exact image ID against pins.json.
ARG BASE=qwen38-int8-lab/eval:0.1.0
FROM ${BASE}
COPY docker/evalplus-requirements.lock /opt/evalplus-requirements.lock
RUN uv venv --python python --no-python-downloads /opt/evalplus && uv pip install --python /opt/evalplus/bin/python --no-cache \
    -r /opt/evalplus-requirements.lock \
    && uv pip check --python /opt/evalplus/bin/python
WORKDIR /app
ENTRYPOINT ["python"]
