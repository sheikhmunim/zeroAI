# syntax=docker/dockerfile:1
#
# Two-stage build. The builder installs dependencies into a self-contained
# virtualenv; the runtime stage copies only that venv. Nothing that exists
# purely to *produce* the environment (pip, wheel caches, build metadata)
# survives into the shipped image.
#
# Base image choice -- the tradeoff:
#   python:3.14          ~1 GB base. Full Debian with compilers and headers.
#                        Convenient, and mostly wasted: every dependency here
#                        ships a prebuilt manylinux wheel, so nothing compiles.
#   python:3.14-slim     ~150 MB base. Debian minus docs, headers, and dev
#                        tooling. Wheels install fine. <- chosen
#   python:3.14-alpine   Smallest, and the wrong answer here. Alpine uses musl
#                        libc; PyTorch publishes no musl wheels, so pip would
#                        fall back to building torch from source. Hours, if it
#                        succeeds at all.
#   distroless           Smaller and more locked-down than slim, but no shell,
#                        which makes `docker exec` debugging impossible. Not
#                        worth it for a service whose image is dominated by
#                        1 GB of torch either way.

# ---------------------------------------------------------------- builder ---
FROM python:3.14-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Requirements are copied and installed BEFORE the source. Docker caches each
# instruction against the files it touched, so editing src/api.py invalidates
# only the layers after this point. If we copied everything first, every source
# edit would reinstall a gigabyte of torch.
COPY requirements-serve.txt .

# The --extra-index-url is doing heavy lifting. Plain `pip install torch` pulls
# the CUDA build and drags in ~2.5 GB of nvidia-* wheels that are dead weight
# on a CPU host. The +cpu variant is roughly a fifth of the size.
RUN pip install --index-url https://pypi.org/simple \
                --extra-index-url https://download.pytorch.org/whl/cpu \
                -r requirements-serve.txt

# Trim what pip leaves behind. Bytecode caches are rebuilt on demand and
# torch ships its own C++ test suite, none of which serves a request. Done in
# the builder so the deletions never appear in a shipped layer at all --
# deleting a file in a later layer hides it but does not reclaim its bytes.
RUN find /opt/venv -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && find /opt/venv -type f -name '*.pyc' -delete \
    && rm -rf /opt/venv/lib/python3.14/site-packages/torch/test \
              /opt/venv/lib/python3.14/site-packages/torch/include \
              /opt/venv/lib/python3.14/site-packages/torch/utils/benchmark

# ---------------------------------------------------------------- runtime ---
FROM python:3.14-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache

COPY --from=builder /opt/venv /opt/venv

# The user is created BEFORE the weights are downloaded, and we switch to it
# first, so the 600 MB of cache files are written already owned by appuser.
#
# The obvious ordering -- download as root, then `chown -R appuser` at the end
# -- cost 606 MB. Changing a file's ownership rewrites it into the new layer,
# and a recursive chown over the weight cache therefore stores a second full
# copy. Layers are immutable: the originals stay in the image underneath.
# Never chmod/chown a large tree after the fact; create it correctly instead.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app /opt/hf-cache \
    && chown appuser:appuser /app /opt/hf-cache

WORKDIR /app
USER appuser

# Bake the CLIP weights into the image rather than fetching them at startup.
#
# The alternative -- download on first boot -- makes the image ~600 MB smaller
# but is the wrong trade for a service:
#   * every cold start pulls 600 MB before it can answer a request, so the
#     health check fails for a minute or more and the platform may kill the
#     container as unhealthy before it ever becomes ready;
#   * a HuggingFace outage or rate-limit becomes a production outage in a
#     service that otherwise has no runtime network dependency;
#   * the running image stops being reproducible -- the same tag could resolve
#     different weights later.
COPY --chown=appuser:appuser src/config.py src/__init__.py ./src/
RUN python -c "\
import open_clip; \
from src.config import CLIP_MODEL, CLIP_PRETRAINED; \
open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED); \
print('cached', CLIP_MODEL, CLIP_PRETRAINED)"

# Set only AFTER the weights are cached -- ENV applies to every following
# instruction, so declaring this in the block above made the download itself
# run offline and fail. Ordering of ENV relative to RUN is load-bearing.
#
# From here on it makes any accidental runtime fetch fail loudly at startup,
# instead of silently working in dev and hanging in production.
ENV HF_HUB_OFFLINE=1

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser frontend/ ./frontend/
COPY --chown=appuser:appuser artifacts/head.pt ./artifacts/head.pt

# Documentation only -- EXPOSE publishes nothing by itself. The actual port is
# whatever -p maps at run time, or $PORT on a PaaS.
EXPOSE 8000

# Hosting platforms (Render, Fly, Cloud Run) inject the port to bind as $PORT
# and will consider the deploy failed if the process ignores it and hardcodes
# 8000. Defaulting keeps `docker run` simple locally.
ENV PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request; \
urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health').read()"

# Exec form (JSON array), not shell form. Shell form wraps the process in
# /bin/sh -c, which does not forward SIGTERM -- so the platform's graceful
# shutdown signal is swallowed and every deploy ends in a 10-second SIGKILL.
# sh -c is still needed here to expand $PORT, so we forward the signal with
# `exec`, which replaces the shell with uvicorn as PID 1.
CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
