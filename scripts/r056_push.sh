#!/usr/bin/env bash
# R056 streaming: every 4 min rebuild label inputs from the cases generated so far, push them atomically to the pod,
# and pull the labels back. Stops when .context/r056/PUSH_STOP exists.  scripts/r056_push.sh <ssh-port> <ip>
set -uo pipefail
cd "$(dirname "$0")/.."
K=~/.runpod/ssh/RunPod-Key-Go; P=$1; H=root@$2; O=(-o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=4); S=(ssh -i $K -p $P "${O[@]}" $H)
until [[ -e .context/r056/PUSH_STOP ]]; do
  python3 scripts/r056_build.py labelin >.context/r056/labelin.last.json
  tar -czf .context/r056/label_in.tgz -C .context/r056 label_in
  scp -q "${O[@]}" -i $K -P $P .context/r056/label_in.tgz $H:/workspace/r054/.context/r056/ &&
    "${S[@]}" 'cd /workspace/r054/.context/r056 && rm -rf label_in.tmp && mkdir label_in.tmp && tar -xzf label_in.tgz -C label_in.tmp &&
      for f in label_in.tmp/label_in/*.jsonl; do mv "$f" label_in/; done'
  mkdir -p .context/r056/labels
  scp -q "${O[@]}" -i $K -P $P "$H:/workspace/g1/r056/labels_*.jsonl" .context/r056/labels/ 2>/dev/null
  echo "$(date -u +%FT%TZ) pushed $(cat .context/r056/labelin.last.json | cut -c1-300)"
  sleep 240
done
