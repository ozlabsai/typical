# R050 frozen external-evaluation runbook

Date: 2026-10-07

Scored work is permitted only after R050 emits one eligible winner. The detached wrapper may be
started earlier in its bounded wait mode. External results must never feed back into checkpoint,
temperature, prompt, loss, or retry selection.

## Frozen authority

The released Benchmark Heaven target is
`https://benchmarkheaven.com/api/jevbench/v1.5.4`. It declares revision `v1.5.4`, protocol
`jevbench::v1.5`, run kind `official`, 1,624 decisions, and Jev 1.13.0 score
`72.13292897895896`. The public 231-row checkout is a diagnostic, not a reproduction of the
sealed official run.

The public JevBench harness is frozen at
`bb05a335bc809e61b20c0f745d25499a82b326fc`. WorkflowEvals is frozen at
`0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb` plus the reviewed Typical provider patch whose
SHA-256 is `07106bf888aa53a2770b83503c58635708db32042ecb4b034a41963b1ba43908`.

Workflow parity covers 705 cases and 1,524 policy decisions. The published Jev gates are:

- security incidents exact action set: at least `0.617`;
- agent trace primary action: at least `0.716`;
- invoice exact action set: at least `0.618`;
- customer service exact action set: at least `0.760`;
- equal-workflow mean: at least `0.678`;
- no missing case, failed request, failed decision, retry, or model substitution.

## Stage private, immutable checkouts

`/workspace` is persistent FUSE storage on the evaluation pod and does not enforce Unix mode
bits. It is suitable for the rehashed model, adapter, pipeline state, and final completed archive,
but not for active code, harnesses, logs, or scored intermediate results. Stage all executable
checkouts on the container's local overlay under `/root/r050-eval`.

Set `EXPECTED_TYPICAL_COMMIT` to the reviewed 40-character deployment commit supplied in the
handoff. Do not discover or derive it from the checkout being launched.

```bash
set -euo pipefail
umask 077

export EXPECTED_TYPICAL_COMMIT='<reviewed-deployment-commit>'
export R050_LOCAL_ROOT=/root/r050-eval
export TYPICAL_REPO=/root/r050-eval/typical
export WORKFLOWEVALS=/root/r050-eval/WorkflowEvals
export JEVBENCH=/root/r050-eval/jevbench

export TYPICAL_SOURCE=/workspace/g1/typical-eval-frozen
export WORKFLOW_SOURCE=/workspace/g1/WorkflowEvals-0ac3b8ad
export JEV_SOURCE=/workspace/g1/jevbench-bb05a335

[[ "$EXPECTED_TYPICAL_COMMIT" =~ ^[0-9a-f]{40}$ ]]
test ! -e "$R050_LOCAL_ROOT"
install -d -m 700 "$R050_LOCAL_ROOT"

git clone --no-hardlinks "$TYPICAL_SOURCE" "$TYPICAL_REPO"
git -C "$TYPICAL_REPO" checkout --detach "$EXPECTED_TYPICAL_COMMIT"
test "$(git -C "$TYPICAL_REPO" rev-parse HEAD)" = "$EXPECTED_TYPICAL_COMMIT"
test -z "$(git -C "$TYPICAL_REPO" status --porcelain=v1 --untracked-files=all)"

git clone --no-hardlinks "$WORKFLOW_SOURCE" "$WORKFLOWEVALS"
git -C "$WORKFLOWEVALS" checkout --detach 0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb
git -C "$WORKFLOWEVALS" apply \
  "$TYPICAL_REPO/patches/workflowevals-typical-provider-0ac3b8ad.patch"

git clone --no-hardlinks "$JEV_SOURCE" "$JEVBENCH"
git -C "$JEVBENCH" checkout --detach bb05a335bc809e61b20c0f745d25499a82b326fc
test -z "$(git -C "$JEVBENCH" status --porcelain=v1 --untracked-files=all)"

# Remove write permission first as an explicit staging gate, then make every
# checkout private. Owner execute bits on directories and scripts are retained.
chmod -R go-w "$TYPICAL_REPO" "$WORKFLOWEVALS" "$JEVBENCH"
chmod -R go-rwx "$TYPICAL_REPO" "$WORKFLOWEVALS" "$JEVBENCH"

cd "$TYPICAL_REPO"
uv sync --frozen --group dev
cd "$WORKFLOWEVALS"
uv sync --frozen
```

JevBench does not carry a `uv.lock` and is data/harness input to the pinned Typical runner; do not
create an evaluator-local lock or environment for it. Every Jev command below runs from the Typical
checkout through Typical's frozen UV environment.

The wrapper scans all regular files and directories in each checkout after pipeline quiescence,
including WorkflowEvals files created by the patch. It excludes only `.git`, virtual environments,
caches, and generated run trees. Any group/other-writable runtime input fails closed.

## Autonomous execution (single normative execution path)

The wrapper is the single normative execution path. It owns one strict-mode process, validates
the checkouts before and after waiting for pipeline quiescence, binds the exact pipeline completion
lineage and winner hashes, warms both harnesses, and starts an authenticated loopback-only model
server. It then reserves the persistent root and claims the adapter immediately before the first
scored request.

All preparation and scored output stays in mode-private local staging. A failed preclaim attempt is
retained under `/root/r050-eval/failed-preclaim` and may be retried at most three times. Once the
adapter claim exists, the wrapper never retries scored work. The persistent evidence root carries
only an unguessable ownership token until postrun verification and both preservation checks pass.
Publication itself is retried at most three times from the receipt-bound private evidence.

```bash
set -euo pipefail
umask 077

export EXPECTED_TYPICAL_COMMIT='<reviewed-deployment-commit>'
export R050_LOCAL_ROOT=/root/r050-eval
export TYPICAL_REPO=/root/r050-eval/typical
export WORKFLOWEVALS=/root/r050-eval/WorkflowEvals
export JEVBENCH=/root/r050-eval/jevbench
export R050_ROOT=/workspace/g1/r050-ab512c3/repo/.context/r050-gpu-run-a77fea5-resume2
export EVAL_ROOT=/workspace/g1/r050-official-eval-20261007
export LAUNCH_LOG=/root/r050-eval/r050-official-eval.launch.log

[[ "$EXPECTED_TYPICAL_COMMIT" =~ ^[0-9a-f]{40}$ ]]
test "$(git -C "$TYPICAL_REPO" rev-parse HEAD)" = "$EXPECTED_TYPICAL_COMMIT"
test -z "$(git -C "$TYPICAL_REPO" status --porcelain=v1 --untracked-files=all)"
test ! -e "$EVAL_ROOT"
test ! -e "$LAUNCH_LOG"
! tmux has-session -t r050-official 2>/dev/null

tmux new-session -d -s r050-official \
  "umask 077; exec env R050_WAIT_FOR_COMPLETE=1 R050_WAIT_TIMEOUT_SECONDS=86400 EXPECTED_TYPICAL_COMMIT=$EXPECTED_TYPICAL_COMMIT R050_LOCAL_ROOT=$R050_LOCAL_ROOT R050_LAUNCH_LOG=$LAUNCH_LOG TYPICAL_REPO=$TYPICAL_REPO R050_ROOT=$R050_ROOT EVAL_ROOT=$EVAL_ROOT WORKFLOWEVALS=$WORKFLOWEVALS JEVBENCH=$JEVBENCH bash $TYPICAL_REPO/scripts/run_r050_official_eval.sh >$LAUNCH_LOG 2>&1"
tmux has-session -t r050-official
```

The `umask 077` inside the detached shell runs before its launch-log redirection. The log therefore
lives on the local overlay with private mode; the wrapper checks this again. After a successful
run it copies the closed launch log into the completed persistent archive and records its SHA-256.

## Safe monitoring

Do not open scored outputs while the run is active. Monitor only process liveness, the launch-log
tail, GPU ownership/utilization, and marker existence:

```bash
tmux has-session -t r050-official
tail -n 30 /root/r050-eval/r050-official-eval.launch.log
nvidia-smi
test -e /workspace/g1/r050-official-eval-20261007/official-eval-complete.json && echo complete
```

The default pipeline-completion wait is bounded at 24 hours. If the pipeline terminates without an
eligible winner, the wrapper exits before loading a model or creating an adapter claim. If the
detached process dies before the claim, retained private evidence can be reviewed and a new wrapper
may be started only after establishing that no claim exists. A failure after the claim invalidates
the official scored attempt and is never permission to rerun scored work.

If all bounded publication attempts fail after scoring and verification succeeded, the wrapper
prints the exact private attempt directory. Recovery is publication-only and must point at that
directory; it revalidates the ready schema, adapter claim, reservation token, run tag, and three
receipt hashes before copying. Run it with the same exported paths and reviewed commit as the
normative launcher:

```bash
export R050_PUBLISH_ONLY_ATTEMPT_ROOT=/root/r050-eval/attempts/<reported-attempt>
R050_PUBLISH_ONLY_ATTEMPT_ROOT="$R050_PUBLISH_ONLY_ATTEMPT_ROOT" \
  EXPECTED_TYPICAL_COMMIT="$EXPECTED_TYPICAL_COMMIT" \
  R050_LOCAL_ROOT="$R050_LOCAL_ROOT" R050_LAUNCH_LOG="$LAUNCH_LOG" \
  TYPICAL_REPO="$TYPICAL_REPO" R050_ROOT="$R050_ROOT" EVAL_ROOT="$EVAL_ROOT" \
  WORKFLOWEVALS="$WORKFLOWEVALS" JEVBENCH="$JEVBENCH" \
  bash "$TYPICAL_REPO/scripts/run_r050_official_eval.sh" >>"$LAUNCH_LOG" 2>&1
```

This path never starts the model server or any benchmark command. If the claim or ownership token
does not exactly match the verified attempt, it fails closed.

## Completion evidence

Success requires `official-eval-complete.json` to be written last. Before publication, the parent
process validates the child's exact ready schema, adapter identity, verification-receipt hash, and
both preservation-receipt hashes. The completed persistent root contains:

- frozen R050 state, completion, winner completion, and their hashes;
- a path-independent relative hash manifest, including the copied Typical `uv.lock`;
- exact Typical, WorkflowEvals, patch, JevBench, model, adapter, runtime, and server identities;
- four complete WorkflowEvals run trees and the 231-row public JevBench diagnostic;
- preflight, postrun, pre-archive, and post-archive preservation receipts;
- the private launch log copy, its SHA-256, and a complete archive checksum manifest;
- the final completion marker written only after every preceding copy and hash succeeds.

Any missing gate or receipt is a failed evaluation, not permission to infer success or rerun scored
requests.
