# VeriDoc AI serving image: CPU-only torch, no checkpoint inside.
#
#   docker build -t veridoc-api:0.5 .
#   docker run --rm -p 8000:8000 \
#     -v "$PWD/artifacts/run2/best.pt:/models/best.pt:ro" \
#     -e VERIDOC_CHECKPOINT=/models/best.pt veridoc-api:0.5
#
# The trained checkpoint is gitignored and is mounted read-only at run time (see README,
# "Getting the checkpoint into the container"). The app refuses to serve if its sha256 does
# not match results/frozen_config_run2.json. No network is needed at serve time.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app \
    VERIDOC_CHECKPOINT=/models/best.pt

WORKDIR /app

# CPU-only torch first. The explicit +cpu versions can only come from the CPU wheel index,
# so no CUDA wheels are pulled in; torch's own dependencies come from PyPI.
RUN pip install torch==2.14.1+cpu torchvision==0.29.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# only what serving needs: the shared inference package, the app, configs and the frozen
# validation config (pixel threshold, score rule, expected checkpoint hash)
COPY src/docforge ./src/docforge
COPY serving ./serving
COPY configs/serve.yaml configs/eval.yaml ./configs/
COPY results/frozen_config_run2.json ./results/

RUN useradd --create-home --uid 10001 veridoc
USER veridoc

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

# one worker: inference is CPU-bound and already multi-threaded inside torch
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
