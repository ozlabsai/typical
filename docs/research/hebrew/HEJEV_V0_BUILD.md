# Hebrew JevBench v0

> **EVAL-ONLY.** Every file here comes from the public JevBench items or was teacher-labeled for this
> benchmark. Never use these items, their translations, their labels or their soft gold for training,
> distillation or training-data selection. The rule is also recorded in `manifest.json` (`eval_only`) and on
> every row (`provenance.eval_only`).

All files use the JevBench task format (`jevbench/tasks.py`), so they run in the **same runner and scorer** as
the English public set. The builder checks every row with `jevbench.tasks.Task.from_dict`.

| file | rows | what |
|---|---|---|
| `translated.jsonl` | 230 | `he-<id>`: Hebrew twins of the 231 public items (original 72, easy 48, hard 111; 1 dropped). Option IDs/labels and gold are unchanged. |
| `mixed.jsonl` | 230 | `mx-<id>`: Hebrew state with the English question and options (the common Israeli enterprise mix). Same gold. |
| `native.jsonl` | 119 | `native-<domain>-<batch>-<i>`: new decisions written in Hebrew for Israeli contexts. Soft gold from two teachers. **Not human-verified.** |
| `native_review_sample.md` | 30 | stratified spot-check list for a native Hebrew speaker |
| `qa.jsonl` | 231 | per-item translation QA: every attempt, deterministic issues, back-translation, judge verdict |
| `manifest.json` | | counts, sha256 per file, QA and labeling stats, notional cost |
| `cache/` | | every raw CLI response, keyed by sha256 of the request. Reruns are free. |

## Construction

Builder: `scripts/hejev_build.py` (`translate`, `native`, `sample`). Scorer: `scripts/hejev_score.py`.
Test: `tests/test_hejev.py`. All model calls go through the `claude -p` transport in
`scripts/r053_teacher_label.py` (`--tools "" --strict-mcp-config --setting-sources "" --no-session-persistence
--output-format json --effort high`, `--json-schema` for structured output, `DISABLE_PROMPT_CACHING=1`).
At most 6 calls run at once.

**Translated twins.** For each item the translatable leaves are the state (string, or every multi-word
string value of a structured state; JSON keys and single-token values such as enums and IDs are left alone), the
instructions, and the criteria values. Labels and criteria keys are never sent. The leaves go to
`claude-fable-5-1` as one JSON object, and a `--json-schema` with exactly the same keys forces the reply into
that shape. The translator prompt asks for natural Israeli-register Hebrew. It requires every number, date,
amount, ID, code, URL, JSON key and option ID to be kept verbatim. Names may be localized only when they are
purely decorative. Every fact, hedge, ambiguity and distractor must be kept, with no hints. QA on each attempt:
1. Deterministic check: every digit run (as a multiset) and every snake_case option ID found in each English
   field must appear in the Hebrew field, and the instructions must contain Hebrew.
2. Back-translation to English by a different model family: `codex exec` with `gpt-5.6-terra`. The fallback
   was `claude-opus-5-5`, but it was never needed: all 233 back-translations came from codex.
3. A judge (`claude-opus-5-5`) gets the English source, the Hebrew, the back-translation, the option IDs and
   the gold answer. It returns `{ok, issues}` and flags meaning drift, changed numbers or labels, answer
   leakage (in either direction) and untranslated or unnatural text.

A flagged item gets one retry, with the issues and the previous translation fed back. If it fails again it is
dropped. Mixed variants reuse the accepted Hebrew state.

**Native items.** `claude-fable-5-1` wrote batches of 12 items per domain: 5 choice, 4 noul and 3 score, with
4 deliberately ambiguous and 8 clear. The domains are חשבוניות מס, מכרזים, שירות לקוחות, אבטחת מידע,
ביטוח/בנקאות, HR and שירותי ממשלה. Batch 0 covers routine cases. Batch 1 covers exceptions, escalations
and conflicting documents. Options use English snake_case IDs with Hebrew descriptions. Each item was then
labeled independently by `claude-fable-5-1` and `claude-opus-5-5` (effort high, one question per call). Both
used WorkflowEvals' `system_one_adapter` probability prompt (`structured_outputs=False`,
`llm_answer_mode="probabilities"`), the same path as `r053_teacher_label.py --backend cli`. The soft gold is
the mean of the two distributions (`provenance.soft_gold`, with the per-teacher distributions alongside), and
`expected` is its argmax. An item is kept only when both teachers have the same argmax. The author's intended
answer and rationale are kept in provenance for reference only, never as gold.

## QA stats

Translation (231 source items):
- First pass flagged: 3/231 (1.3%). The deterministic check flagged 1 (a missing digit in the longest hard
  item). The judge flagged 2.
- Retried: 3. Fixed by retry: 2. Dropped: 1, `hard-sol-a-multi_hop-07`. In that item the English lets a reader
  resolve "RA" from the initials of "returning associate". The first Hebrew lost that link (harder). The retry
  added "(RA)" (leakage). The item has no faithful Hebrew rendering, so it was dropped.
- Accepted: 230 translated and 230 mixed (original 72, easy 48, hard 110).
- Caveat: a 1.3% judge flag rate is low. The judge's flags were specific and correct (an alias that only works
  in English; the word חליפה too close to the gold option's החלפה), but it is one judge. The native-speaker
  sample covers native items only.

Native items: 132 generated in 11 batches: batch 0 for all 7 domains, batch 1 for invoices, tenders, customer
service and infosec. All 132 were well-formed. The budget cap stopped labeling, so the 12 items of infosec
batch 1 were never labeled. 120 got labels from both teachers.
- Teacher argmax agreement: **119/120** (clear 79/79, ambiguous 40/41; choice 51/51, noul 39/40, score 29/29).
  The mean JSD between the two teachers is 0.0034.
- Kept: **119** (choice 51, noul 39, score 29; clear 79, ambiguous 40). By domain: invoices 24, tenders 23,
  customer service 23, infosec 13, insurance/banking 12, HR 12, government 12. The soft-gold argmax matches
  the author's intended answer on 115/119.
- Caveat: the two teachers are both Claude models and agree almost perfectly, even on the items written to be
  "ambiguous". Those items are borderline by design, but the teachers do not see them as ambiguous. Read the
  soft gold as Claude consensus, not as human uncertainty. Gold is teacher-labeled and **not
  human-verified**: see `native_review_sample.md` (30 items: 10 per type, 15 clear and 15 ambiguous, 4-5
  per domain).

## Cost (notional CLI `total_cost_usd`, cap $60)

| stage | $ |
|---|---|
| Part 1 + 2 translation: fable translator (234 calls, 3 of them retries) + opus judge (233), including the 5-item pilot | **41.80** |
| Part 3 native generation (fable, 11 batches) | 13.43 |
| Part 3 native labels (fable + opus, one question per call) | 4.87 |
| **total** (cap $60; in-flight calls overshot it by $0.10) | **60.10** |

By model, from `cache/`: claude-fable-5-1 $51.39 over 373 calls, claude-opus-5-5 $8.71 over 355 calls, codex
233 calls.

Codex back-translation ran on the ChatGPT subscription and is counted as $0 notional.

## How to run a model on it

These are drop-in task files for the stock runner. The runner refuses result paths inside the jevbench repo,
so write results elsewhere. Example with the TypeSafe `/v1/systemone` adapter, as in
`scripts/r054_pod_phase0.sh` and `.context/run_g0.sh`:

```bash
H=$PWD/.context/hebrew-bench/hejev-v0; R=$PWD/.context/hebrew-bench/runs/<model>
cd .context/jevbench-v154
for s in translated mixed native; do
  uv run python -m jevbench.cli run --tasks $H/$s.jsonl --adapter typesafe --endpoint http://127.0.0.1:8010 \
    --key-env '' --model <served-name> --cost-basis self_hosted_gpu --reserve-usd 0 \
    --results $R/$s.jsonl --raw-dir $R/raw-$s --ledger $R/ledger-$s.jsonl --manifest $R/manifest-$s.json \
    --run-label hejev-$s --delay-s 0
done
cd -
uv run python scripts/hejev_score.py --en <English run on the public 231: file or dir> \
  --he $R/translated.jsonl --mx $R/mixed.jsonl --native $R/native.jsonl --json $R/score.json
```

`r054_pod_phase0.sh`'s `run_jev` only shards files named `easy|original|hard*`. For a sharded pod run, split
these files the same way, or call the CLI directly as shown above.

The scorer reports, per split: coverage, accuracy (argmax, invalid counts as wrong), valid rate, Brier
(multi-class sum, binary 2-class convention, from `jevbench.metrics`) and top-label ECE (10 bins). For twins
(`he-`/`mx-` vs English, overall and per tier) it reports the accuracy gap (HE−EN), the flip counts in each
direction, prediction agreement, and the mean Jensen-Shannon divergence (log2, 0..1) between the EN and HE
probability vectors. On the existing English runs it reproduces R054's counts: Cygnet 203/231 (.879,
Brier .171), Gemma-4-12B think 225/231 (.974), Qwen3.8-27B think 229/231 (.991). No Hebrew run exists yet.
