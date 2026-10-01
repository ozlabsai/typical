#!/bin/bash
# One RunPod GPU pod hosting typically for invited users, reachable through a Cloudflare quick tunnel (deploy/README.md).
#   deploy/runpod.sh up      build the app, create the pod, copy the repo + deploy/hosted.env, start it, print the public URL
#   deploy/runpod.sh url     print the public URL again
#   deploy/runpod.sh logs    tail the server log
#   deploy/runpod.sh down    copy the data dir back to .context/typically-hosted-<time>.tgz, then delete the pod
# Needs locally: runpodctl (configured), ssh/scp, git, node/npm, python3. Secrets live in deploy/hosted.env and are never printed
# or passed on a command line: the file is scp'd (0600) to the pod.
# IDLE GUARD: the pod bills by the hour whether anyone uses it or not. Run `down` when the stunt is over; the training pods it
# rents are deleted by the job runner itself (and swept at server start), but this host pod is only ever deleted by `down`.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=typically-host
IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404   # same image as the training pods (scripts/typically_job.py)
GPUS=("NVIDIA L4" "NVIDIA RTX A5000")
KEY="$HOME/.runpod/ssh/RunPod-Key-Go"
STATE=deploy/.pod   # the pod id of the running host
ENV_FILE=deploy/hosted.env
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20)

json() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }
pod_id() { [ -f "$STATE" ] && cat "$STATE" || { echo "no host pod recorded in $STATE" >&2; exit 1; }; }
addr() {   # "<ip> <port>" of the pod's ssh
  runpodctl pod get "$(pod_id)" -o json | json "(d.get('ssh') or {}).get('ip','') + ' ' + str((d.get('ssh') or {}).get('port',''))"
}
on_pod() { read -r ip port < <(addr); ssh -n "${SSH_OPTS[@]}" -p "$port" "root@$ip" "$@"; }

up() {
  [ -f "$STATE" ] && { echo "a host pod is already recorded ($(cat "$STATE")); run 'down' first" >&2; exit 1; }
  [ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE (copy deploy/hosted.env.example)" >&2; exit 1; }
  for v in TYPICALLY_SECRET TYPICALLY_INVITES; do grep -q "^$v=." "$ENV_FILE" || { echo "$ENV_FILE needs $v" >&2; exit 1; }; done
  (cd typically-app && npm ci && npm run build)
  tmp=$(mktemp -d); tar="$tmp/repo.tgz"
  # what the job runner ships, plus the built app; .context/, .venv/, node_modules/, deploy/hosted.env are git-ignored
  git ls-files -co --exclude-standard | grep -v -E '^(figures|paper|blog|docs|vendor)/' | while read -r f; do [ ! -f "$f" ] || echo "$f"; done > "$tar.list"
  find site/app -type f >> "$tar.list"
  tar czf "$tar" -T "$tar.list"
  echo "repo tarball: $(du -h "$tar" | cut -f1)"

  pub=$(cat "$KEY.pub")
  for gpu in "${GPUS[@]}"; do
    if out=$(runpodctl pod create --name "$NAME" --image "$IMAGE" --gpu-id "$gpu" --cloud-type SECURE \
               --container-disk-in-gb 120 --ports 22/tcp --env "$(python3 -c 'import json,sys; print(json.dumps({"PUBLIC_KEY": sys.argv[1]}))' "$pub")" 2>&1) \
       && id=$(echo "$out" | json "d['id']" 2>/dev/null); then
      echo "$id" > "$STATE"; echo "pod $id ($gpu) created"; break
    fi
    echo "no $gpu available: ${out:0:200}"
  done
  [ -f "$STATE" ] || { echo "no GPU available; try again later" >&2; exit 1; }

  for _ in $(seq 60); do   # ssh up (about 1-3 minutes)
    read -r ip port < <(addr) || true
    [ -n "${port:-}" ] && ssh -n "${SSH_OPTS[@]}" -p "$port" "root@$ip" true 2>/dev/null && break
    sleep 10
  done
  on_pod "mkdir -p /workspace/typically"
  scp "${SSH_OPTS[@]}" -P "$port" "$tar" "root@$ip:/workspace/repo.tgz"
  scp "${SSH_OPTS[@]}" -P "$port" "$ENV_FILE" "root@$ip:/workspace/hosted.env"
  rm -rf "$tmp"
  on_pod "chmod 600 /workspace/hosted.env && tar xzf /workspace/repo.tgz -C /workspace/typically && \
          setsid nohup bash /workspace/typically/deploy/start.sh > /workspace/serve.log 2>&1 < /dev/null &"
  echo "starting (installing the env takes a few minutes)..."
  for _ in $(seq 90); do
    if u=$(on_pod "cat /workspace/public_url 2>/dev/null") && [ -n "$u" ]; then
      echo "public URL: $u/app/   (share links: $u/s/...)"; return
    fi
    sleep 10
  done
  echo "no URL yet; check: deploy/runpod.sh logs" >&2
}

down() {
  id=$(pod_id)
  backup=".context/typically-hosted-$(date +%Y%m%d-%H%M%S).tgz"
  mkdir -p .context
  # the users' models and uploads live only on the pod: keep a copy before it is gone
  if on_pod "tar czf - -C /workspace typically-data" > "$backup" 2>/dev/null && [ -s "$backup" ]; then
    echo "data saved to $backup"
  else
    rm -f "$backup"
    read -r -p "could not copy the data back; delete the pod anyway? [y/N] " ok
    [ "$ok" = y ] || exit 1
  fi
  runpodctl pod delete "$id"
  rm -f "$STATE"
  echo "pod $id deleted"
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  url) on_pod "cat /workspace/public_url" ;;
  logs) on_pod "tail -n 100 /workspace/serve.log" ;;
  *) echo "usage: deploy/runpod.sh up|url|logs|down" >&2; exit 2 ;;
esac
