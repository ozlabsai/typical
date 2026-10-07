# WorkflowEvals Typical provider patch

This directory contains a disclosed integration patch for the upstream
[`typesafe-ai/WorkflowEvals`](https://github.com/typesafe-ai/WorkflowEvals)
repository.

- Required upstream commit: `0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb`
- Patch: `workflowevals-typical-provider-0ac3b8ad.patch`
- Patch SHA-256: `622c550c33a78f4452d7e961d75269fb93674d3ef110edd990c16b495f179bf7`
- Patch size: 16,321 bytes, 351 lines

The patch adds an honestly identified `typical` provider using the native
`TypeSafeClient` protocol. It keeps TypeSafe's Jev-only model restriction,
requires a dedicated Typical endpoint and credential, passes an immutable
literal model identifier, rejects response model substitution, defaults to
joint questions with thinking off, and deliberately adds no pricing claim.
It also adds eight standard-library unit tests covering those contracts and
configuration serialization.

## Reproduce verification

Start from a clean checkout at the exact commit above. Replace `/path/to/typical`
with this repository's absolute path.

```bash
git rev-parse HEAD
git status --porcelain
git apply --check /path/to/typical/patches/workflowevals-typical-provider-0ac3b8ad.patch
git apply /path/to/typical/patches/workflowevals-typical-provider-0ac3b8ad.patch
uv run --frozen python -m unittest discover -v
uv run --frozen python -m compileall -q core run.py tests/test_typical_provider.py
git diff --check
```

The first command must print the required upstream commit, and the second must
print nothing. The test command must run eight passing tests. `uv run --frozen`
must leave `uv.lock` unchanged.

## Integrity constraints

- Invoke Typical as `--model typical:<literal-immutable-model>`.
- Supply either `--base-url` or `TYPICAL_BASE_URL`; there is no fallback to the
  public TypeSafe endpoint. Supply the host root (for example,
  `http://127.0.0.1:8090`), not a URL ending in `/v1`; the SDK appends the API
  path.
- Supply `TYPICAL_API_KEY`. The provider never reads or falls back to
  `TYPESAFE_API_KEY`.
- The endpoint must echo the requested literal model exactly in
  `response.model`; a mismatch fails the call.
- Do not add a `typical:*` entry to `core/pricing.json` without measured,
  reviewable pricing evidence.

No workflow benchmark calls were made while producing or verifying this patch.
