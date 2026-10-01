# typically, hosted with sign-in. Build from the repo root:
#   docker build -f deploy/Dockerfile -t typically .                       # CPU torch (small image; slow model answers)
#   docker build -f deploy/Dockerfile --build-arg TORCH=cu130 -t typically .   # CUDA torch (run with --gpus all)
# Run: docker run -p 8787:8787 --env-file deploy/hosted.env -v typically-data:/data typically   (then put TLS in front: deploy/README.md)

FROM node:22-slim AS app
WORKDIR /src/typically-app
COPY typically-app/package.json typically-app/package-lock.json ./
RUN npm ci
COPY typically-app/ ./
RUN npm run build   # vite writes ../site/app

FROM python:3.12-slim
# TORCH: a download.pytorch.org/whl index (cpu, cu128, cu130); TORCH_VERSION: the lock's. A CUDA 12.8 driver host needs
# TORCH=cu128 TORCH_VERSION=2.11.0 (the same rule as scripts/typically_spike_pod.sh).
ARG TORCH=cpu
ARG TORCH_VERSION=2.14.0
ARG TARGETARCH=amd64
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# git: the job runner snapshots the repo with `git ls-files`; ssh/scp + runpodctl: it rents and drives the training pods
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client curl ca-certificates && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL -o /usr/local/bin/runpodctl "https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-${TARGETARCH}" \
 && chmod +x /usr/local/bin/runpodctl
WORKDIR /app
COPY pyproject.toml uv.lock ./
# every locked dep except the CUDA stack (--no-deps: the export is the whole closure), then torch from the chosen wheel index
RUN uv export --frozen --no-hashes --no-emit-project -o /tmp/lock.txt \
 && grep -v -E '^(torch|triton|nvidia-|cuda-)' /tmp/lock.txt > /tmp/requirements.txt \
 && uv venv .venv --python 3.12 \
 && uv pip install --python .venv/bin/python --no-deps -r /tmp/requirements.txt \
 && uv pip install --python .venv/bin/python "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH}"
COPY . .
COPY --from=app /src/site/app site/app
ENV TYPICALLY_DATA=/data HOST=0.0.0.0 PORT=8787 PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8787
CMD ["bash", "deploy/serve.sh"]
