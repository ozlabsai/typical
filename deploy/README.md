# Hosting typically for invited users

A small, invite-only deployment: sign in with an invite code, see and train only your own models (plus the
Northwind sample and the base models), a few training runs per day, and an opt-in public results card per model.
With `TYPICALLY_AUTH` unset, the local dev server behaves exactly as before (no sign-in, no quotas).

What sign-in on (`TYPICALLY_AUTH=1`, `site/typically_auth.py`) changes:

- Every `/api/*` call needs a session, except `/api/auth/*` and `/api/health`. `/v1/models/{id}/decide` keeps its own
  API keys. `/s/*` share pages, the static site and the `/app` bundle stay public (the app shows the sign-in screen itself).
- Models, uploads, corrections, API keys and share links belong to whoever made them. Admins see everything.
- Quotas: `TYPICALLY_DAILY_RUNS` (3) training runs per user per rolling 24 h (admins exempt), one training job per user
  at a time, `TYPICALLY_MAX_JOBS` (2) jobs on the whole server. They apply to "Teach it" and to retrain.
- No CORS: the app and the API are served from the same origin.

## Settings

| Variable | |
|---|---|
| `TYPICALLY_AUTH=1` | sign-in on (`deploy/serve.sh` sets it) |
| `TYPICALLY_SECRET` | signs session cookies (HMAC-SHA256). Required; the server refuses to start without it |
| `TYPICALLY_INVITES` | invite codes, comma-separated. More can go in `<data>/invites.json` (a JSON list) |
| `TYPICALLY_ADMINS` | admin codes (also valid invites) |
| `TYPICALLY_DAILY_RUNS`, `TYPICALLY_MAX_JOBS` | quotas, defaults 3 and 2 |
| `TYPICALLY_DATA` | data dir (jobs, results, uploads, keys, shares); default `.context/typically`, `/workspace/typically-data` on the pod, `/data` in Docker |
| `TYPICALLY_PUBLIC_URL` | base URL for share links; default: the request's host |
| `TYPICALLY_POD_PREFIX` | training pod name prefix; `serve.sh` uses `typically-hosted-job-` so a hosted server and a local one on the same RunPod account never sweep each other's pods |
| `RUNPOD_API_KEY` | training pods (runpodctl 2.x reads it from the env). `serve.sh` creates the `~/.runpod/ssh/RunPod-Key-Go` key pair the job runner hands each training pod |
| `HF_TOKEN` | the training data / base checkpoints on Hugging Face (read by `scripts/typically_job.py`, copied to each training pod as a 0600 file) |
| `OPENROUTER_API_KEY` | optional: "Read by Claude" plans and question parsing |

Invite codes are the only credential, and there is no login rate limit, so make them long and random:
`python3 -c "import secrets; print(secrets.token_urlsafe(12))"`. A user's id is a hash of their code, so a given code always
maps to the same models.

## Run locally with sign-in

```bash
cd typically-app && npm run build && cd ..
TYPICALLY_AUTH=1 TYPICALLY_SECRET=dev-secret TYPICALLY_INVITES=try-me-123 TYPICALLY_ADMINS=admin-456 \
  uv run uvicorn --app-dir site server:app --port 8787
# open http://127.0.0.1:8787/app/ and sign in with try-me-123
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
job tracking and quotas live in the server process.

## RunPod: one GPU pod behind a Cloudflare tunnel

```bash
cp deploy/hosted.env.example deploy/hosted.env    # fill it in; it is git-ignored
deploy/runpod.sh up      # builds the app, creates the pod (L4, else A5000, 3090, A40, A6000; secure cloud), prints the https URL
deploy/runpod.sh url     # the URL again
deploy/runpod.sh logs    # the server log
deploy/runpod.sh down    # copies the data back to .context/typically-hosted-<time>.tgz, then deletes the pod
```

`up` copies the repo (what `git ls-files` lists, minus figures/paper/blog/docs/vendor, plus the built app) and
`hosted.env` (scp, 0600, never on a command line) to the pod and runs `deploy/start.sh` there: it installs the env, runpodctl
and cloudflared, opens a quick tunnel, and starts `deploy/serve.sh` on 127.0.0.1 (no port is open on the pod; only the tunnel
reaches it). The pod has the runpodctl key, so its "Teach it" rents H100 training pods exactly like the local server does.

The quick tunnel's URL is random and changes when the pod restarts. For a stable address, create a named tunnel on your
Cloudflare account (`cloudflared tunnel create typically`, route a hostname to it), run it on the pod with its token instead of
`--url`, and set `TYPICALLY_PUBLIC_URL` to that hostname. If RunPod restarts the pod, run `deploy/start.sh` again over ssh.

**Idle guard.** The host pod bills by the hour whether anyone uses it or not, and nothing deletes it but `down`. Put a
calendar reminder on the stunt's end date, and check `runpodctl pod list` afterwards. Training pods are deleted by the
job runner when each run ends (and orphans are swept when the server starts).

## Invites, admins, secret

- **Add an invite:** append to `TYPICALLY_INVITES` and restart, or add it to `<data>/invites.json` on the pod (no restart:
  it is read on every request).
- **Revoke an invite:** remove it. Its sessions stop working on their next request; its models stay on disk (admins still see them).
- **Rotate the secret:** change `TYPICALLY_SECRET` and restart. Everyone is signed out; nobody loses models (ids come from
  the invite code, not the secret).

## Costs (check current RunPod prices)

- Host pod: an L4, RTX A5000, RTX 3090, A40 or RTX A6000 on secure cloud, roughly $0.30-0.50 an hour, about $8-12 a day while it is up.
- Training: one H100 per run, about $2.50-3 an hour. A small-base run takes roughly 25-45 minutes (about $1-2), a
  medium-base run about twice that. Worst case per day = invited users x `TYPICALLY_DAILY_RUNS` runs, and at most
  `TYPICALLY_MAX_JOBS` H100s at once.
- Cloudflare quick tunnels are free.
