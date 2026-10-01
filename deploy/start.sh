#!/bin/bash
# Runs ON the RunPod host pod (deploy/runpod.sh up copies the repo to /workspace/typically and starts this detached).
# Installs the Python env + cloudflared, opens a Cloudflare quick tunnel to 127.0.0.1:8787, then runs deploy/serve.sh.
# Secrets come from /workspace/hosted.env (0600, copied by runpod.sh); nothing here prints them.
set -euo pipefail
cd /workspace/typically
set -a; . /workspace/hosted.env; set +a
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen
# same driver rule as scripts/typically_spike_pod.sh: a CUDA 12.8 driver needs the cu128 torch build
if [[ "$(nvidia-smi)" == *"CUDA Version: 12.8"* ]]; then
  uv pip install --python .venv/bin/python "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
fi
command -v runpodctl >/dev/null || { curl -fsSL -o /usr/local/bin/runpodctl https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 && chmod +x /usr/local/bin/runpodctl; }
command -v cloudflared >/dev/null || { curl -fsSL -o /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /usr/local/bin/cloudflared; }

# quick tunnel: a new random https://*.trycloudflare.com URL per start (a named tunnel gives a stable one: deploy/README.md)
nohup cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8787 > /workspace/tunnel.log 2>&1 &
for _ in $(seq 60); do
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /workspace/tunnel.log | head -1 || true)
  [ -n "$URL" ] && break
  sleep 2
done
[ -n "${URL:-}" ] || { echo "cloudflared did not print a URL; see /workspace/tunnel.log" >&2; exit 1; }
echo "$URL" > /workspace/public_url
echo "PUBLIC_URL $URL"
export TYPICALLY_PUBLIC_URL="${TYPICALLY_PUBLIC_URL:-$URL}"   # absolute links in share cards
HOST=127.0.0.1 exec bash deploy/serve.sh   # only the tunnel reaches it: no open port on the pod
