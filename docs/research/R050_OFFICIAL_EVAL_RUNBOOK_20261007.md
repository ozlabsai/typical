# R050 frozen external-evaluation runbook

Date: 2026-10-07

This runbook is executable only after R050 emits one eligible winner. The same frozen base,
adapter, option-logit readout, and runtime settings are used for every external evaluation.
External results must never feed back into checkpoint, temperature, prompt, or loss selection.

## 0. Frozen targets and authority

The external target is the released Benchmark Heaven API artifact at
`https://benchmarkheaven.com/api/jevbench/v1.5.4`, not the repository README's older
"current" label. On 2026-10-07 that artifact declares:

- revision `v1.5.4`, protocol `jevbench::v1.5`, status `released`, run kind `official`;
- source SHA-256 `452885de2a84cd5b9ed393d541fd8d6c6a540f9d7f762f9d2df9349383f26173`;
- 904 open plus 720 sealed decisions, 1,624 total;
- frozen method SHA-256 `c25d3d8b8512e4d93370a9e0c99705d19b2a9389956ca33b8a4bd2b0ec501c07`;
- official equal-axis/equal-type headline addendum SHA-256
  `752ddccc4e191c81412631ee3121563421c4b9077cb6c3f2243ed1d1545cfab0`;
- Jev 1.13.0 full coverage, 1,624 answered, zero missing, score
  `72.13292897895896`, and Intelligence `71.99974990450261`.

The public `fstandhartinger/jevbench` checkout is separately pinned at commit
`bb05a335bc809e61b20c0f745d25499a82b326fc`. It contains the frozen v1.5 method but its
README still calls v1.4.2.2 current. Its 231 public rows are therefore a legacy diagnostic,
not a reproduction of the 1,624-row official v1.5.4 run.

The frozen method originally proposed a 40/20/20/20 headline and 50/25/25 type weights. The
disclosed post-leaderboard addendum makes official v1.5 headline A use equal Intelligence,
Calibration, Speed, and Cost axes and equal Choice, Noul, and Score weights. Parity is always
computed under that released amended headline, never the superseded proposal.

WorkflowEvals parity means all 705 cases and 1,524 policy decisions at the pinned dataset
revisions below, with these published Jev gates:

- security incidents exact action set: at least `0.617`;
- agent trace primary action: at least `0.716`;
- invoice exact action set: at least `0.618`;
- customer service exact action set: at least `0.760`;
- equal-workflow mean: at least `0.678`;
- no missing case, failed request, failed decision, retry, or model substitution.

## 1. Winner and artifact gate

Use a new evaluation directory and derive every artifact from the sealed pipeline state. Never
type a base-model or winner path by hand.

```bash
set -euo pipefail
umask 077

export TYPICAL_REPO=/workspace/g1/r050-ab512c3/repo
export R050_ROOT=/workspace/g1/r050-ab512c3/repo/.context/r050-gpu-run-a77fea5-resume2
export EVAL_ROOT=/workspace/g1/r050-official-eval-20261007
export WORKFLOWEVALS=/workspace/g1/WorkflowEvals-0ac3b8ad
export JEVBENCH=/workspace/g1/jevbench-bb05a335

STATE="$R050_ROOT/state.json"
COMPLETE="$R050_ROOT/complete.json"
test "$(jq -r .status "$COMPLETE")" = complete
test "$(jq -c .official_evaluations_run "$COMPLETE")" = '[]'

WINNER=$(jq -er .winner "$COMPLETE")
jq -e --arg winner "$WINNER" '.eligible_candidates | index($winner) != null' "$COMPLETE"
ARM_COMPLETION="$R050_ROOT/runs/$WINNER/completion.json"
ADAPTER=$(jq -er .artifacts.checkpoint.path "$ARM_COMPLETION")
ADAPTER_SHA=$(jq -er .artifacts.checkpoint.tree_sha256 "$ARM_COMPLETION")
ADAPTER_FILES=$(jq -er .artifacts.checkpoint.files "$ARM_COMPLETION")

BASE_MODEL=$(jq -er .binding.paths.model "$STATE")
MODEL_INVENTORY=$(jq -er .binding.paths.model_inventory "$STATE")
MODEL_INVENTORY_SHA=$(jq -er .binding.hashes.model_inventory "$STATE")
BASE_MODEL_SHA=$(jq -er .tree_sha256 "$MODEL_INVENTORY")
BASE_REVISION=$(jq -er .revision "$MODEL_INVENTORY")
MODEL_ID="typical-r050-sha256-${ADAPTER_SHA:0:16}"

test ! -e "$EVAL_ROOT"
mkdir -p "$EVAL_ROOT"

curl --fail --location --silent --show-error \
  https://benchmarkheaven.com/api/jevbench/v1.5.4 \
  --output "$EVAL_ROOT/jevbench-v1.5.4-target.json"

cd "$TYPICAL_REPO"
uv run --frozen python - "$EVAL_ROOT/jevbench-v1.5.4-target.json" <<'PY'
import json
import math
import sys
from pathlib import Path

target = json.loads(Path(sys.argv[1]).read_text())
assert (target["revision"], target["protocol"], target["status"], target["run_kind"]) == (
    "v1.5.4", "jevbench::v1.5", "released", "official"
)
assert target["source_sha256"] == "452885de2a84cd5b9ed393d541fd8d6c6a540f9d7f762f9d2df9349383f26173"
assert target["method_sha256"] == "c25d3d8b8512e4d93370a9e0c99705d19b2a9389956ca33b8a4bd2b0ec501c07"
assert target["headline_method_addendum_sha256"] == "752ddccc4e191c81412631ee3121563421c4b9077cb6c3f2243ed1d1545cfab0"
assert target["sample"] == {"open": 904, "sealed": 720, "total": 1624, "published_open": 601}
assert target["headline"] == "A" and set(target["types"].values()) == {1 / 3}
jev = next(row for row in target["systems"] if row["key"] == "jev-1.13.0")
assert jev["status"] == {"status": "complete", "rows": 1624, "missing": 0, "answered_ok": 1624}
assert math.isclose(jev["jevbench_score"], 72.13292897895896, rel_tol=0, abs_tol=1e-12)
assert math.isclose(jev["axes"]["intelligence"], 71.99974990450261, rel_tol=0, abs_tol=1e-12)
PY
```

Authenticate the full base and adapter trees before loading weights:

```bash
cd "$TYPICAL_REPO"
uv run --frozen python - "$MODEL_INVENTORY" "$BASE_MODEL" \
  "$MODEL_INVENTORY_SHA" "$ADAPTER" "$ADAPTER_SHA" "$ADAPTER_FILES" <<'PY'
import sys
from pathlib import Path
from scripts.build_r047_model_inventory import verify_inventory
from scripts.preflight_r043_assets import tree_hash

inventory, inventory_sha = verify_inventory(Path(sys.argv[1]), Path(sys.argv[2]))
assert inventory_sha == sys.argv[3]
adapter_sha, adapter_files = tree_hash(Path(sys.argv[4]))
assert adapter_sha == sys.argv[5]
assert adapter_files == int(sys.argv[6])
print(inventory["tree_sha256"], adapter_sha)
PY
```

Before observing any benchmark result, archive `state.json`, `complete.json`, the winner
completion receipt, their SHA-256 hashes, the repository commit, and `uv.lock` hash.

## 2. Start the frozen TypeSafe-compatible server

The SDK base URL is the host root; never append `/v1`. Use one private key and one server receipt
for the complete WorkflowEvals run.

Start it in the same shell that will run the harness. Keeping the key only in the shell
environment avoids copying it into another terminal or exposing it on a command line.

```bash
export TYPICAL_API_KEY="$(openssl rand -hex 32)"
export TYPICAL_BASE_URL=http://127.0.0.1:8090

cd "$TYPICAL_REPO"
uv run --frozen python scripts/serve_typesafe_gemma_lora.py \
  --model "$BASE_MODEL" \
  --model-revision "$BASE_REVISION" \
  --expected-model-tree-sha256 "$BASE_MODEL_SHA" \
  --adapter "$ADAPTER" \
  --expected-adapter-tree-sha256 "$ADAPTER_SHA" \
  --model-id "$MODEL_ID" \
  --receipt "$EVAL_ROOT/server-startup-receipt.json" \
  --max-context 16384 \
  --dtype bfloat16 \
  --device cuda \
  --attn-implementation sdpa \
  --host 127.0.0.1 \
  --port 8090 >"$EVAL_ROOT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

while test ! -s "$EVAL_ROOT/server-startup-receipt.json"; do
  kill -0 "$SERVER_PID"
  sleep 5
done

curl --fail --silent --show-error \
  --header "Authorization: Bearer $TYPICAL_API_KEY" \
  "$TYPICAL_BASE_URL/v1/models" \
  | jq -e --arg model "$MODEL_ID" '.models | length == 1 and .[0].name == $model'
```

The server refuses an unauthenticated non-loopback bind. Its immutable receipt must bind the exact
base and adapter canonical hashes, model ID, runtime settings, and server/runtime/scorer code.

## 3. Prepare the disclosed WorkflowEvals harness

Apply the tracked patch only to a clean checkout at the exact upstream commit. The patch gives
Typical a distinct provider and credential, uses the native TypeSafe protocol, checks the echoed
model ID, and sets `RetryPolicy(max_retries=0)` so failures cannot be hidden.

```bash
PATCH="$TYPICAL_REPO/patches/workflowevals-typical-provider-0ac3b8ad.patch"
test "$(sha256sum "$PATCH" | cut -d' ' -f1)" = \
  07106bf888aa53a2770b83503c58635708db32042ecb4b034a41963b1ba43908
test "$(git -C "$WORKFLOWEVALS" rev-parse HEAD)" = \
  0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb
test -z "$(git -C "$WORKFLOWEVALS" status --porcelain)"
git -C "$WORKFLOWEVALS" apply --check "$PATCH"
git -C "$WORKFLOWEVALS" apply "$PATCH"

cd "$WORKFLOWEVALS"
uv lock --check
uv run --frozen python -m unittest discover -v
```

The frozen verification suite must pass 9/9. The harness must record provider `typical`, lab
`other`, exact model ID, native probabilities, zero retries, and no claimed price. Never route
Typical through `typesafe:jev-1.13.0` and never reuse `TYPESAFE_API_KEY`.

Freeze all server, harness, dataset, task, patch, dependency, and code identities before the first
scored request. This receipt is immutable and the post-run verifier independently recomputes the
same identities; a self-edited receipt cannot authorize drift.

```bash
cd "$TYPICAL_REPO"
uv run --frozen python scripts/verify_r050_official_eval.py preflight \
  --startup-receipt "$EVAL_ROOT/server-startup-receipt.json" \
  --workflowevals-dir "$WORKFLOWEVALS" \
  --workflow-patch "$PATCH" \
  --jevbench-dir "$JEVBENCH" \
  --jev-runner "$TYPICAL_REPO/scripts/jevbench_gemma_lora.py" \
  --task-file "$JEVBENCH/datasets/public/easy.jsonl" \
  --task-file "$JEVBENCH/datasets/public/original.jsonl" \
  --task-file "$JEVBENCH/datasets/public/hard.jsonl" \
  --base-url "$TYPICAL_BASE_URL" \
  --timeout 1800 \
  --out "$EVAL_ROOT/preflight-receipt.json"
```

## 4. Run WorkflowEvals exactly once

Use a unique run tag and prove none of the four output directories exists. `--questions all`
matches the Jev methodology. Both worker counts remain one because the correctness-first server
keeps a case's multi-question request contiguous; invoice requests can contain 120 questions.

```bash
cd "$WORKFLOWEVALS"
RUN_TAG="typical-r050-${ADAPTER_SHA:0:16}-$(date -u +%Y%m%dT%H%M%SZ)"
for workflow in security_incidents agent_trace_observability invoice_processing customer_service; do
  test ! -e "$WORKFLOWEVALS/runs/$workflow/$RUN_TAG"
done

uv run --frozen python run.py security_incidents \
  --model "typical:$MODEL_ID" --name "$RUN_TAG" \
  --base-url "$TYPICAL_BASE_URL" --workers 1 --question-workers 1 \
  --questions all --timeout 1800 \
  --dataset-revision fbe1ea5c69cf494157fd23f2002a0d9d9a418443

uv run --frozen python run.py agent_trace_observability \
  --model "typical:$MODEL_ID" --name "$RUN_TAG" \
  --base-url "$TYPICAL_BASE_URL" --workers 1 --question-workers 1 \
  --questions all --timeout 1800 \
  --dataset-revision 8635540973910a92465fe2bc53e195375aa6e1a8

uv run --frozen python run.py invoice_processing \
  --model "typical:$MODEL_ID" --name "$RUN_TAG" \
  --base-url "$TYPICAL_BASE_URL" --workers 1 --question-workers 1 \
  --questions all --timeout 1800 \
  --dataset-revision 6beeb2d2acd65c086c835022f5f4d7434114cafc

uv run --frozen python run.py customer_service \
  --model "typical:$MODEL_ID" --name "$RUN_TAG" \
  --base-url "$TYPICAL_BASE_URL" --workers 1 --question-workers 1 \
  --questions all --timeout 1800 \
  --dataset-revision b1342f5a704587dbc465867c38c2694348ff86e4
```

Do not use `--limit` or `--retry-errors`. A documented infrastructure failure may be followed by
a new, separately named run; never resume or overwrite a scored candidate after observing output.

## 5. Run the 231-row public JevBench diagnostic

Stop the HTTP server first so a second 12B model cannot be loaded beside it. Confirm that it exits;
do not start the diagnostic while the server still owns GPU memory. This command uses the
same base, adapter, dtype, context, and native option-logit readout, and binds local artifacts to
the exact canonical R050 hashes before model load:

```bash
kill "$SERVER_PID"
wait "$SERVER_PID" || true
trap - EXIT

cd "$TYPICAL_REPO"
test ! -e "$EVAL_ROOT/jevbench-public-231"
test "$(git -C "$JEVBENCH" rev-parse HEAD)" = \
  bb05a335bc809e61b20c0f745d25499a82b326fc
test -z "$(git -C "$JEVBENCH" status --porcelain)"

uv run --frozen python scripts/jevbench_gemma_lora.py \
  --jevbench-dir "$JEVBENCH" \
  --task-file "$JEVBENCH/datasets/public/easy.jsonl" \
  --task-file "$JEVBENCH/datasets/public/original.jsonl" \
  --task-file "$JEVBENCH/datasets/public/hard.jsonl" \
  --model "$BASE_MODEL" \
  --model-revision "$BASE_REVISION" \
  --expected-model-tree-sha256 "$BASE_MODEL_SHA" \
  --adapter "$ADAPTER" \
  --expected-adapter-tree-sha256 "$ADAPTER_SHA" \
  --out "$EVAL_ROOT/jevbench-public-231" \
  --max-context 16384 \
  --dtype bfloat16 \
  --device cuda \
  --attn-implementation sdpa \
  --seed 42
```

The 231-row output is diagnostic evidence only. It cannot establish v1.5.4 parity and must not
change the model. Official parity requires the external evaluator's fixed 1,624-row run with all
720 sealed decisions, zero missing/invalid outputs, score at least `72.13292897895896`, and
Intelligence at least `71.99974990450261`.

## 6. Verify and preserve evidence

Run the committed verifier after all four workflow directories and the Jev diagnostic are closed.
It schema-validates the upstream WorkflowEvals bundles and recomputes their scores, then replays
Jev's frozen per-task scoring from the pinned tasks and native probabilities. It also validates
exact IDs/counts, failure and retry absence, immutable model/config identity, startup receipt,
input hashes, harness commit, runner code, raw records, and the reservation ledger before any claim
is reported.

```bash
cd "$TYPICAL_REPO"
uv run --frozen python scripts/verify_r050_official_eval.py postrun \
  --preflight "$EVAL_ROOT/preflight-receipt.json" \
  --startup-receipt "$EVAL_ROOT/server-startup-receipt.json" \
  --workflowevals-dir "$WORKFLOWEVALS" \
  --workflow-run "$WORKFLOWEVALS/runs/security_incidents/$RUN_TAG" \
  --workflow-run "$WORKFLOWEVALS/runs/agent_trace_observability/$RUN_TAG" \
  --workflow-run "$WORKFLOWEVALS/runs/invoice_processing/$RUN_TAG" \
  --workflow-run "$WORKFLOWEVALS/runs/customer_service/$RUN_TAG" \
  --jevbench-dir "$JEVBENCH" \
  --jev-runner "$TYPICAL_REPO/scripts/jevbench_gemma_lora.py" \
  --jev-run "$EVAL_ROOT/jevbench-public-231" \
  --out "$EVAL_ROOT/verification-receipt.json"

jq -e '.integrity_pass == true' "$EVAL_ROOT/verification-receipt.json"

uv run --frozen python scripts/verify_r050_official_eval.py preserved \
  --verification "$EVAL_ROOT/verification-receipt.json" \
  --workflow-run "$WORKFLOWEVALS/runs/security_incidents/$RUN_TAG" \
  --workflow-run "$WORKFLOWEVALS/runs/agent_trace_observability/$RUN_TAG" \
  --workflow-run "$WORKFLOWEVALS/runs/invoice_processing/$RUN_TAG" \
  --workflow-run "$WORKFLOWEVALS/runs/customer_service/$RUN_TAG" \
  --jev-run "$EVAL_ROOT/jevbench-public-231" \
  --out "$EVAL_ROOT/preservation-receipt.json"

jq -e '.evidence_preserved == true' "$EVAL_ROOT/preservation-receipt.json"
```

`integrity_pass` proves that the evidence is authentic and complete; it does not itself assert
benchmark parity. Workflow parity is reported separately in `.workflow_parity`, while official
JevBench v1.5.4 parity remains `unmeasured` until the external 1,624-row evaluator returns it.
The receipt is an internal, content-addressed consistency proof, not hardware-backed remote
attestation: a cryptographic claim about physical model execution requires the external evaluator
to sign or append-only commit its run. Do not describe the local receipt as that stronger claim.
Repeat the `preserved` check after archival and immediately before reporting results. It binds the
whole run trees, so later byte edits or added files invalidate the preservation receipt.

Archive the JSON verification and preservation receipts alongside:

- winner, pipeline, model-inventory, and startup receipts;
- repository commit and UV lock hash;
- WorkflowEvals upstream commit, patch hash, four manifests/results/scores, and dataset revisions;
- JevBench harness commit, task-file hashes, raw records, ledger, and manifest;
- complete commands, stdout/stderr, latency, token usage, GPU and cost telemetry;
- every procedural deviation, including the already logged R050 operator blinding deviation.

The final integrity audit is run once after these artifacts exist. Any missing evidence is a failed
gate, not permission to infer success.
