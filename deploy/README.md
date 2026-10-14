# Hosting Decision Factory for invited users

A small, invite-only deployment: sign in with an invite code, see and train only your own models (plus the
Northwind sample and the standard models), a few training runs per day, and an opt-in public results card per model.
With `TYPICALLY_AUTH` unset, the local dev server behaves exactly as before (no sign-in, no quotas).

What sign-in on (`TYPICALLY_AUTH=1`, `site/typically_auth.py`) changes:

- Every `/api/*` call needs a session, except `/api/auth/*` and `/api/health`. `/v1/models/{id}/decide` keeps its own
  API keys. `/s/*` share pages, the static site and the `/app` bundle stay public (the app shows the sign-in screen itself).
- Models, uploads, corrections, API keys and share links belong to whoever made them. Admins see everything.
- Quotas: `TYPICALLY_DAILY_RUNS` (3) training runs per user per rolling 24 h (admins exempt, and failed runs count too), one
  training job per user at a time, `TYPICALLY_MAX_JOBS` (2) jobs on the whole server. They apply to "Start training" and to retrain.
- Rate limits on the endpoints that call the LLM key on every request: 30 table analyses and 120 chat question parses per user
  per hour (`typically_auth.LIMITS`, kept in memory and reset on restart).
- Hugging Face push is admin-only, always private, and the repo name is chosen by the server (it uses the operator's token).
  Everyone else keeps "Download weights".
- No CORS: the app and the API are served from the same origin.

## Settings

| Variable | |
|---|---|
| `TYPICALLY_AUTH=1` | sign-in on (`deploy/serve.sh` sets it) |
| `TYPICALLY_SECRET` | signs session cookies (HMAC-SHA256). Required: the server refuses to start without it |
| `TYPICALLY_INVITES` | invite codes, comma-separated. More can go in `<data>/invites.json` (a JSON list) |
| `TYPICALLY_ADMINS` | admin codes (also valid invites) |
| `TYPICALLY_DAILY_RUNS`, `TYPICALLY_MAX_JOBS` | quotas, defaults 3 and 2 |
| `TYPICALLY_DATA` | data dir (jobs, results, uploads, keys, shares). Default `.context/typically`, `/workspace/typically-data` on the pod, `/data` in Docker |
| `TYPICALLY_PUBLIC_URL` | base URL for share links. Default: the request's host |
| `TYPICALLY_POD_PREFIX` | training pod name prefix. `serve.sh` uses `typically-hosted-job-` so a hosted server and a local one on the same RunPod account never sweep each other's pods |
| `RUNPOD_API_KEY` | training pods (runpodctl 2.x reads it from the env). `serve.sh` creates the `~/.runpod/ssh/RunPod-Key-Go` key pair the job runner hands each training pod |
| `HF_TOKEN` | the training data / base checkpoints on Hugging Face (read by `scripts/typically_job.py`, copied to each training pod as a 0600 file) |
| `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY` | optional (Anthropic wins if both): AI reading of uploads, question parsing in chat, and the Enrich step's extra examples and translations. No spend cap (see Costs). Without a key the built-in rules are used and those two Enrich options are off |

Invite codes are the only credential, and there is no login rate limit, so make them long and random:
`python3 -c "import secrets; print(secrets.token_urlsafe(12))"`. A user's id is a hash of their code, so a given code always
maps to the same models.

## Run locally with sign-in

```bash
cd typically-app && npm run build && cd ..
TYPICALLY_AUTH=1 TYPICALLY_SECRET=dev-secret TYPICALLY_INVITES=try-me-123 TYPICALLY_ADMINS=admin-456 \
  uv run uvicorn --app-dir site server:app --port 8787
# open http://127.0.0.1:8787/ and sign in with try-me-123 (with sign-in on, the app is the whole site)
```

On localhost the cookie is not marked `Secure`, so plain http works. Anywhere else it is, so serve over https.

## Docker

```bash
docker build -f deploy/Dockerfile -t typically .                    # CPU torch
docker build -f deploy/Dockerfile --build-arg TORCH=cu130 -t typically .   # CUDA (lock's torch 2.14)
# a host whose driver is CUDA 12.8: --build-arg TORCH=cu128 --build-arg TORCH_VERSION=2.11.0
# building on Apple Silicon for an x86 host: add --platform linux/amd64
docker run -p 8787:8787 --env-file deploy/hosted.env -v typically-data:/data typically
```

The container listens on 0.0.0.0:8787 with sign-in on. Put TLS in front of it (a Cloudflare tunnel, Caddy, your platform's
proxy), and set `FORWARDED_ALLOW_IPS` to the proxy's address so share links and the cookie see `https`. Run one container:
job tracking and quotas live in the server process. Stop it with `docker stop -t 120`: on shutdown the server deletes the
training GPUs of runs in flight (those runs fail and do not resume), which can take up to ~90 s.

The Northwind sample model is git-ignored. The build copies it from `.context/typically/results/co_f/` and
`data_co_a/eval/a_oneliner.jsonl` when they are in the build context, and `deploy/serve.sh` seeds it into `/data`. Without
them the sample still lists, but its chat, results and download do not work.

## RunPod: one GPU pod behind a Cloudflare tunnel

```bash
cp deploy/hosted.env.example deploy/hosted.env    # fill it in; it is git-ignored
deploy/runpod.sh up      # builds the app, creates the pod (L4, else A5000, 3090, A40, A6000; secure cloud), prints the https URL
deploy/runpod.sh url     # the URL again
deploy/runpod.sh logs    # the server log
deploy/runpod.sh down    # stops the server, copies the data back to .context/typically-hosted-<time>.tgz, deletes its training pods, then the pod
```

If anything fails after the pod is created, or the server is not healthy within 15 minutes, `up` prints the server log,
deletes the pod and exits non-zero.

`up` copies the repo (what `git ls-files` lists, minus figures/paper/blog/docs/vendor, plus the built app and the Northwind
sample model, see Docker above) and
`hosted.env` (scp, 0600, never on a command line) to the pod and runs `deploy/start.sh` there: it installs the env, runpodctl
and cloudflared, opens a quick tunnel, and starts `deploy/serve.sh` on 127.0.0.1 (no port is open on the pod, and only the tunnel
reaches it). The pod has the runpodctl key, so its "Start training" rents H100 training pods exactly like the local server does.

The quick tunnel's URL is random and changes when the pod restarts. For a stable address, create a named tunnel on your
Cloudflare account (`cloudflared tunnel create typically`, route a hostname to it), run it on the pod with its token instead of
`--url`, and set `TYPICALLY_PUBLIC_URL` to that hostname. If RunPod restarts the pod, run `deploy/start.sh` again over ssh.

**Idle guard.** The host pod bills by the hour whether anyone uses it or not, and nothing deletes it but `down`. Put a
calendar reminder on the date the invite period ends, and check `runpodctl pod list` afterwards. Training pods are deleted by
the job runner when each run ends, by the server's shutdown, and by a sweep every 5 minutes (a pod whose run has not reported
for 15 minutes). `down` deletes any `typically-hosted-job-*` pod left.

## GCP: one L4 VM behind a named Cloudflare tunnel (decisions.ozlabs.ai)

The live deployment. `deploy/gcp.sh` (project `ozla-476923` unless `GCP_PROJECT` is set) runs the Docker image above with CUDA torch
on a `g2-standard-4` VM (1x L4, on-demand, 150 GB pd-balanced boot disk, Deep Learning VM image with driver 580). The VM sits in its
own VPC whose only ingress rule is ssh from Google's IAP range: the app listens on 127.0.0.1 and only the `cloudflared` container
on the same host reaches it. Data lives on the boot disk (`/var/lib/decision-factory/data`, mounted as `/data`); secrets in
`/etc/decision-factory/` (root, 0600).

One-time, on a Mac whose `cloudflared` is logged in to the ozlabs.ai zone:

```bash
cloudflared tunnel create decision-factory          # writes ~/.cloudflared/<tunnel-id>.json
# --config: an empty file, or cloudflared routes to the tunnel named in ~/.cloudflared/config.yml instead of the one given
: > /tmp/empty.yml && cloudflared tunnel --config /tmp/empty.yml route dns <tunnel-id> decisions.ozlabs.ai
deploy/gcp.sh create     # VPC + IAP-only firewall + Artifact Registry repo + the VM (first L4 zone in stock, europe-west4 first)
deploy/gcp.sh build      # Cloud Build (amd64, cu130), pushes europe-west4-docker.pkg.dev/<project>/decision-factory/typically
cp deploy/hosted.env.example deploy/hosted.env   # fill it in; TYPICALLY_PUBLIC_URL=https://decisions.ozlabs.ai
deploy/gcp.sh deploy ~/.cloudflared/<tunnel-id>.json && rm deploy/hosted.env   # the VM keeps the only copy
```

Then: `deploy/gcp.sh build && deploy/gcp.sh deploy` to redeploy (the env and tunnel credentials on the VM are kept),
`deploy/gcp.sh logs`, `deploy/gcp.sh stop` / `start` (a stopped VM bills only its disk), `deploy/gcp.sh down` to tear it all down
(copies the data back to `.context/` first). Add an invite without a restart: `deploy/gcp.sh ssh`, then edit
`/var/lib/decision-factory/data/invites.json` (a JSON list).

Cost (europe-west1 list prices): about $0.78 an hour for the VM with its L4, plus the disk (about $15 a month) and the external IP;
about $590 a month if left on. Training pods are billed on RunPod as below.

## Invites, admins, secret

- **Add an invite:** append to `TYPICALLY_INVITES` and restart, or add it to `<data>/invites.json` on the pod (no restart:
  it is read on every request).
- **Users invite too:** every signed-in user can make up to 5 active invite codes (admins: unlimited) from "Invite people" in
  the app; they live in `<data>/issued_invites.json` ([{code, by, created}]). Revoking one there (owner or admin) frees the slot.
- **Revoke an invite:** remove it. Its sessions stop working on their next request. Its models stay on disk (admins still see them).
- **Rotate the secret:** change `TYPICALLY_SECRET` and restart. Everyone is signed out, and nobody loses models (ids come from
  the invite code, not the secret).

## Costs (check current RunPod prices)

- Host pod: an L4, RTX A5000, RTX 3090, A40 or RTX A6000 on secure cloud, roughly $0.30-0.50 an hour, about $8-12 a day while it is up.
- Training: one H100 per run (about $2.70 an hour). Measured: a small-base run takes about 10 minutes (8-14, one outlier at
  28), about $0.60. A medium-base run about 25 minutes, about $1.50. Thorough (800 steps) roughly doubles the training part.
  Worst case per day = invited users x `TYPICALLY_DAILY_RUNS` runs, and at most `TYPICALLY_MAX_JOBS` H100s at once.
- AI enrichment (Enrich's extra examples and translations, at most 3 languages per run) runs on the operator's LLM key with
  no spend cap. On a large table with several languages one run can cost tens of dollars.
- Cloudflare quick tunnels are free.
