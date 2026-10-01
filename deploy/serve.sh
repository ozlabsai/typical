#!/bin/bash
# The hosted typically server: sign-in on (site/typically_auth.py), one process (job tracking and quotas are in-process).
# Used as the Docker CMD and by deploy/start.sh on the RunPod host. Put TLS in front of it (the Cloudflare tunnel, or your proxy).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TYPICALLY_SECRET:?set TYPICALLY_SECRET (see deploy/README.md)}"
export TYPICALLY_AUTH=1
export TYPICALLY_DATA="${TYPICALLY_DATA:-/workspace/typically-data}"
# its training pods get their own name prefix, so this server's orphan sweep never deletes a local dev server's pods (same account)
export TYPICALLY_POD_PREFIX="${TYPICALLY_POD_PREFIX:-typically-hosted-job-}"
mkdir -p "$TYPICALLY_DATA"
# the training runner rents GPUs with runpodctl; `config` stores the key and creates the ~/.runpod/ssh key typically_job.py uses
if [ -n "${RUNPOD_API_KEY:-}" ] && [ ! -f "$HOME/.runpod/ssh/RunPod-Key-Go" ]; then
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null
fi
[ -d .git ] || git init -q   # typically_job snapshots the repo with `git ls-files -co --exclude-standard`; untracked files are enough
exec "${PYTHON:-.venv/bin/python}" -m uvicorn --app-dir site server:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8787}" \
  --proxy-headers --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-127.0.0.1}"
