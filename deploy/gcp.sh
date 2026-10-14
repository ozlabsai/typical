#!/bin/bash
# Decision Factory on one GCP GPU VM (g2-standard-4, 1x L4) behind a Cloudflare named tunnel (deploy/README.md, "GCP").
#   deploy/gcp.sh create            one-time: a VPC whose only ingress is IAP ssh, the VM, the Artifact Registry repo
#   deploy/gcp.sh build             Cloud Build the CUDA image (repo + built app + Northwind sample) and push it
#   deploy/gcp.sh deploy [CREDS]    copy deploy/hosted.env (if present locally) and the tunnel credentials JSON (if given) to the VM
#                                   (0600, root), pull the image, (re)start the app and cloudflared containers
#   deploy/gcp.sh logs | ssh        the server log | a shell on the VM (IAP)
#   deploy/gcp.sh stop | start      stop the server cleanly (its shutdown deletes its training GPUs), then the VM | start the VM again
#   deploy/gcp.sh down              stop the server, copy the data dir back to .context/, sweep training pods, delete the VM, network, repo
# Secrets are never printed or passed on a command line: files go over `gcloud compute scp`. The tunnel and its DNS record are
# made once on the Mac (README). Every gcloud call names the project; the global gcloud config is never changed.
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT=${GCP_PROJECT:-ozla-476923}
REGION=${GCP_REGION:-europe-west4}   # the image registry + Cloud Build
# the VM: the first zone with an L4 in stock, closest to Israel first (me-west1 has no L4)
ZONES=${GCP_ZONES:-"europe-west4-a europe-west4-b europe-west4-c europe-west3-a europe-west3-b europe-west6-b europe-west6-c europe-west1-b europe-west1-c europe-west2-a europe-west2-b"}
VM=decision-factory
NET=decision-factory-net
REPO=decision-factory
IMAGE=$REGION-docker.pkg.dev/$PROJECT/$REPO/typically:latest
ENV_FILE=deploy/hosted.env
G=(--project "$PROJECT")

zone() { gcloud compute instances list "${G[@]}" --filter="name=$VM" --format='value(zone.basename())'; }
on_vm() { gcloud compute ssh "$VM" "${G[@]}" --zone "$(zone)" --tunnel-through-iap --quiet --command "$1"; }
to_vm() { gcloud compute scp "${G[@]}" --zone "$(zone)" --tunnel-through-iap --quiet "$1" "$VM:$2"; }

create() {
  # idempotent: a re-run (after a stockout) picks up where the last one stopped
  gcloud compute networks describe "$NET" "${G[@]}" >/dev/null 2>&1 || gcloud compute networks create "$NET" "${G[@]}" --subnet-mode=custom
  # the only ingress: ssh from Google's IAP range. The app is reached through the tunnel (outbound), never a port
  gcloud compute firewall-rules describe "$NET-iap-ssh" "${G[@]}" >/dev/null 2>&1 ||
    gcloud compute firewall-rules create "$NET-iap-ssh" "${G[@]}" --network "$NET" --direction INGRESS --allow tcp:22 \
    --source-ranges 35.235.240.0/20
  gcloud artifacts repositories describe "$REPO" "${G[@]}" --location "$REGION" >/dev/null 2>&1 ||
    gcloud artifacts repositories create "$REPO" "${G[@]}" --repository-format docker --location "$REGION"
  i=0
  for z in $ZONES; do   # L4 stock comes and goes per zone
    r=${z%-*}; i=$((i + 1))
    gcloud compute networks subnets describe "$NET-$r" "${G[@]}" --region "$r" >/dev/null 2>&1 ||
      gcloud compute networks subnets create "$NET-$r" "${G[@]}" --network "$NET" --region "$r" --range "10.80.$i.0/24"
    gcloud compute instances create "$VM" "${G[@]}" --zone "$z" --machine-type g2-standard-4 \
      --maintenance-policy TERMINATE --restart-on-failure --subnet "$NET-$r" \
      --image-project deeplearning-platform-release --image-family common-cu129-ubuntu-2404-nvidia-580 \
      --boot-disk-size 150GB --boot-disk-type pd-balanced --metadata block-project-ssh-keys=true 2>&1 | grep -E 'Created|STOCKOUT|unavailable|ERROR' && [ -n "$(zone)" ] && break
  done
  zone >/dev/null && [ -n "$(zone)" ] || { echo "no L4 in stock in $ZONES right now; try later" >&2; exit 1; }
  for _ in $(seq 30); do on_vm true 2>/dev/null && break; sleep 10; done
  # docker + the NVIDIA container toolkit (the DL image's driver is 580 = CUDA 13: the image's cu130 torch fits)
  on_vm "set -e; command -v docker >/dev/null || curl -fsSL https://get.docker.com | sudo sh
    dpkg -s nvidia-container-toolkit >/dev/null 2>&1 || { curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-ct.gpg
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-ct.gpg] https://#' | sudo tee /etc/apt/sources.list.d/nvidia-ct.list >/dev/null
      sudo apt-get update -q && sudo apt-get install -yq nvidia-container-toolkit; }
    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker && sudo systemctl enable docker
    sudo gcloud auth configure-docker $REGION-docker.pkg.dev --quiet
    sudo mkdir -p -m 700 /etc/decision-factory && sudo mkdir -p /var/lib/decision-factory/data && nvidia-smi -L"
}

build() {
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  # the same file set deploy/runpod.sh ships (deploy/Dockerfile.dockerignore trims it further), plus the git-ignored Northwind sample
  { git ls-files -co --exclude-standard | grep -v -E '^(figures|paper|blog|docs|vendor)/' | while read -r f; do [ ! -f "$f" ] || echo "$f"; done
    for f in .context/typically/results/co_f/{best.pt,reveal.json,eval_co.json} data_co_a/eval/a_oneliner.jsonl; do
      [ -f "$f" ] && echo "$f" || echo "warning: $f missing; the Northwind sample will not work" >&2; done; } > "$tmp/list"
  mkdir "$tmp/ctx" && COPYFILE_DISABLE=1 tar --no-xattrs -cf - -T "$tmp/list" | tar -xf - -C "$tmp/ctx"
  : > "$tmp/ctx/.gcloudignore"   # upload all of it (gcloud would otherwise apply .gitignore and drop the sample)
  cat > "$tmp/cloudbuild.yaml" <<EOF
steps:
- name: gcr.io/cloud-builders/docker
  env: [DOCKER_BUILDKIT=1]
  args: [build, -f, deploy/Dockerfile, --build-arg, TORCH=cu130, -t, '$IMAGE', .]
images: ['$IMAGE']
options: {machineType: E2_HIGHCPU_8}
timeout: 3600s
EOF
  gcloud builds submit "$tmp/ctx" "${G[@]}" --region "$REGION" --config "$tmp/cloudbuild.yaml"
}

deploy() {
  if [ -f "$ENV_FILE" ]; then to_vm "$ENV_FILE" /tmp/hosted.env && on_vm "sudo install -m 600 -o root -g root /tmp/hosted.env /etc/decision-factory/hosted.env && rm /tmp/hosted.env"; fi
  if [ -n "${1:-}" ]; then to_vm "$1" /tmp/tunnel.json && on_vm "sudo install -m 600 -o root -g root /tmp/tunnel.json /etc/decision-factory/tunnel.json && rm /tmp/tunnel.json"; fi
  # host network + 127.0.0.1: only cloudflared (same host) reaches the server. stop -t 120: its shutdown deletes its training GPUs
  on_vm "set -e; sudo docker pull -q $IMAGE
    sudo docker stop -t 120 typically 2>/dev/null || true; sudo docker rm typically 2>/dev/null || true
    sudo docker run -d --name typically --restart always --gpus all --network host --stop-timeout 120 \
      --env-file /etc/decision-factory/hosted.env -e HOST=127.0.0.1 -v /var/lib/decision-factory/data:/data $IMAGE
    # by id (a name needs the account cert); the image runs as uid 65532, so that uid owns the 0600 credentials
    id=\$(sudo python3 -c 'import json; print(json.load(open(\"/etc/decision-factory/tunnel.json\"))[\"TunnelID\"])')
    sudo chown 65532 /etc/decision-factory/tunnel.json; sudo docker rm -f cloudflared 2>/dev/null || true
    sudo docker run -d --name cloudflared --restart always --network host -v /etc/decision-factory/tunnel.json:/etc/tunnel.json:ro \
      cloudflare/cloudflared:latest tunnel --no-autoupdate run --credentials-file /etc/tunnel.json --url http://127.0.0.1:8787 \$id
    for _ in \$(seq 60); do curl -fs -m 5 -o /dev/null http://127.0.0.1:8787/api/health && echo healthy && exit 0; sleep 5; done
    echo 'not healthy after 5 minutes'; sudo docker logs --tail 40 typically; exit 1"
}

stop() { on_vm "sudo docker stop -t 120 typically" || true; gcloud compute instances stop "$VM" "${G[@]}" --zone "$(zone)"; }

down() {
  gcloud compute instances start "$VM" "${G[@]}" --zone "$(zone)" --quiet   # running or not: the data lives on its disk
  for _ in $(seq 30); do on_vm "sudo docker stop -t 120 typically" && break; sleep 10; done   # its shutdown deletes its training GPUs
  backup=".context/decision-factory-$(date +%Y%m%d-%H%M%S).tgz"; mkdir -p .context
  on_vm "sudo tar czf - -C /var/lib/decision-factory data" > "$backup" && [ -s "$backup" ] && echo "data saved to $backup" ||
    { read -r -p "could not copy the data back; delete the VM anyway? [y/N] " ok; [ "$ok" = y ] || exit 1; }
  runpodctl pod list -o json | python3 -c 'import json,sys; [print(p["id"]) for p in json.load(sys.stdin) if (p.get("name") or "").startswith("typically-hosted-job-")]' |
    while read -r p; do runpodctl pod delete "$p" && echo "training pod $p deleted"; done
  gcloud compute instances delete "$VM" "${G[@]}" --zone "$(zone)" --quiet
  gcloud compute firewall-rules delete "$NET-iap-ssh" "${G[@]}" --quiet
  gcloud compute networks subnets list "${G[@]}" --network "$NET" --format='value(name,region.basename())' |
    while read -r s r; do gcloud compute networks subnets delete "$s" "${G[@]}" --region "$r" --quiet; done
  gcloud compute networks delete "$NET" "${G[@]}" --quiet
  gcloud artifacts repositories delete "$REPO" "${G[@]}" --location "$REGION" --quiet
  echo "the tunnel and decisions.ozlabs.ai stay: cloudflared tunnel delete decision-factory, and remove the DNS record in the dashboard"
}

case "${1:-}" in
  create) create ;;
  build) build ;;
  deploy) deploy "${2:-}" ;;
  logs) on_vm "sudo docker logs --tail 100 typically 2>&1" ;;
  ssh) gcloud compute ssh "$VM" "${G[@]}" --zone "$(zone)" --tunnel-through-iap ;;
  stop) stop ;;
  start) gcloud compute instances start "$VM" "${G[@]}" --zone "$(zone)" ;;
  down) down ;;
  *) echo "usage: deploy/gcp.sh create|build|deploy [tunnel.json]|logs|ssh|stop|start|down" >&2; exit 2 ;;
esac
