# R051 coverage-shift preregistration

Date: 2026-10-07

Status: conditional; written before R050 was unblinded. R051 runs only if R050 does not produce a
candidate that clears the frozen internal, Hebrew, multimodal, WorkflowEvals, and official
JevBench gates. No R050 arm outcome was inspected while selecting this hypothesis.

## Decision

The next highest-information intervention is a source-diverse, long-context, full-policy data
curriculum with unchanged protected Hebrew and multimodal replay. It is not an RL stage, an
attention retrofit, an activation change, or a new optimizer.

The causal hypothesis is:

> Typical's remaining workflow gap is dominated by narrow semantic/template coverage and weak
> long-context evidence binding, while the current executable mixture repeatedly renders too few
> unique policy groups. Increasing independently generated semantic coverage and natural dispersed
> evidence will improve complete action sets more than changing the learning algorithm or backbone.

This is consistent with the disclosed [Strands Decider history](https://github.com/strands-labs/strands-decider/blob/main/research/history.md), where real-document, multistep, teacher-supported data and replay mattered more than raw synthetic volume. It is also consistent with the fact that the leading [Cygnet recipe](https://github.com/blockbrain-ai/cygnet-recipe) retains frozen Gemma-4-12B-it attention. Neither observation proves the hypothesis; the matched experiment below is the test.

## Why other directions are conditional

1. **RLCD or setwise RL.** R050's typed outcome distribution is enumerable and its verified gold
   is known. Exact NLL/Brier therefore uses every action probability without sampling variance.
   The reported [OpenJev-RLCD](https://arxiv.org/abs/2609.38850) result motivates RL when reward is
   sequential or nondifferentiable, but does not show an advantage over a matched exact supervised
   objective for this setting.
2. **Answer-order training.** Both orders remain a mandatory robustness control, but R050 already
   contains counterfactual views. [PriDe](https://arxiv.org/abs/2309.03882) establishes that option
   position can bias multiple choice; removing that bias does not by itself establish additional
   semantic correctness.
3. **Kimi Delta Attention.** [Kimi Linear](https://arxiv.org/abs/2510.26692) validates KDA as a
   natively pretrained long-context efficiency architecture, not as a cheap competence retrofit.
   The current frozen full-attention Gemma backbone is already represented among JevBench leaders.
   KDA remains a serving-scale branch after quality freezes.
4. **Optimizer or activation changes.** No current evidence identifies optimizer saturation or an
   activation bottleneck. Broad changes risk erasing pretrained Hebrew and visual representations.
   They require a failed oracle-fit or localized mechanistic diagnostic first.

## Matched Coverage-Shift A/B

Both arms use the same parent checkpoint, LoRA architecture, optimizer, schedule, update count,
labeled-outcome count, and total non-padding token budget. The parent and objective are selected
only by the predeclared R050 internal gates.

### Implementation feasibility audit

The current R050 mixer cannot satisfy this design by changing its sample count alone. It requires
eight rendered rows per executable group, so 1,600 executable slots still contain only 200 unique
groups. Its `exact_policy_v1` schema is also Boolean-only and cannot represent a single complete
policy over balanced Choice, Noul, and Score questions.

The underlying exact grouped loss remains reusable. R051 introduces a narrow `typed_policy_v1`
layer that maps each categorical field to a declared policy event:

```text
event_logit = logsumexp(logits[positive labels])
            - logsumexp(logits[complement labels])
```

Those event logits feed the existing verified Boolean policy compiler and exact outcome
NLL/Brier. This supports categorical Choice and ordinal Score fields without replacing the
executor or serving readout. Every declared event must be a nonempty proper subset of its field's
labels, and every field must remain outcome-causal under the compiled policy.

The target executable construction is exactly 800 semantic roots with two answer-order siblings:
792 five-question roots plus eight four-question roots, totaling 3,992 typed decisions per
canonical presentation and 7,984 after both orders. Type totals are fixed at 2,662 Choice, 2,662
Noul, and 2,660 Score decisions. Noul truth values and ordinal ranks are balanced. Siblings share
state, targets, policy, lineage, and split; only option insertion order differs.

Current generated long rows do not meet the declared natural 8k–12k-token requirement, and R050's
8,192-token cap cannot admit the top of that bucket. R051 therefore requires newly pinned,
license-cleared natural source snapshots with upstream IDs/revisions and a 16,384-token training
limit. Five percent of semantic roots (40/800) form the first long-context bucket, with decisive
evidence placed across early, middle, and late token quantiles. Removing the decisive exhibit must
flip the intended field or complete policy outcome.

### Arm A — regenerated current-distribution control

- Regenerate the R050-v2 sampling schedule from its frozen source pools.
- Keep 50% executable, 25% broad decision, 12.5% natural Hebrew, and 12.5% real-image slots.
- Keep field CE plus the R050-selected complete grouped-outcome loss.

### Arm B — coverage-shift treatment

Keep Arm A's broad, Hebrew, and real-image protected slots byte-identical. Replace only the
executable half with:

- at least four times as many unique semantic groups and fewer cosmetic views per group;
- complete workflow/policy episodes rather than isolated fields;
- natural long documents with dispersed decisive evidence and a declared 8k–12k-token bucket;
- balanced Choice, Noul, and Score coverage, including explicit Noul polarity and ordinal
  distributions;
- training-only hard negatives selected from parent errors, capped by source and semantic group;
- teacher probabilities only when independently executable or human gold validates teacher top-1;
- both answer orders, with siblings kept in the same split;
- source-, domain-, generator-, and policy-template-disjoint validation.

A target construction is 3,200 logical slots: 1,600 executable slots from roughly 800 unique
groups with two answer-order views per group, 800 broad slots, 400 Hebrew slots, and 400
multimodal slots. If long sequences prevent matching raw row counts, match non-padding tokens and
labeled outcomes instead. No benchmark item, output, or score enters either arm.

Arm B copies the 800 broad, 400 Hebrew, and 400 multimodal raw JSONL lines from the frozen R050 mix
at the same slot positions; only executable blocks may differ. The builder must bind and recheck
the source and ordered-payload hashes rather than regenerating these protected rows. Image bytes
are copied into a new tree and rehashed; mutable hard links are forbidden.

Frozen protected identities:

| Artifact | SHA-256 |
|---|---|
| R050 training mix | `499dd9c9a8341e55d1d965bbd1ffc45c893880fcdab4d35d295d06324c1c2003` |
| R050 mix manifest | `2b52da6fac16a803d5dd2fa14bcc54c68693f01754b3039561386602f287b53d` |
| Hebrew replay source | `9b840968df4b8a4c9999f231e2e41324b6f02dd8852d6ce2ca2e8e458f5b1d97` |
| Multimodal replay source | `c3097083caa6ea57996222de51e0dd0d021460860d23ea8e00d155aadfe7f50e` |
| 400-image tree | `fe221501247fc3e41f0d48df5339b99a602636806a7375204827f46de1f147c9` |
| Ordered broad payload | `6d63c21eb820ea9ce40c60399ba85ee4fbfc7c6e73e9791de61d99c095f638c2` |
| Ordered Hebrew payload | `483407e28701af1d2149451098736ba4583a1e9d62092826ea26aee61021c2d1` |
| Ordered multimodal payload | `2203502d8421d299d4b561c4ad81a1dfd959ea82157a4ed8e500da2f7117f0d5` |

Training-only hard-root selection scores one canonical view for every newly generated candidate
root using the frozen parent. It ranks wrong complete outcomes first, then complete-outcome NLL,
mean gold-label NLL, and semantic-group ID. Select 200 hard roots and 600 deterministically
stratified roots, capped at 20 per source-family/domain/policy-template cell and 80 per source
family. Add both option-order siblings only after root selection. The score receipt binds the
parent model/adapter trees, candidate-pool hash, scorer code, runtime, and exact root inventory; it
rejects benchmark paths or IDs, missing/duplicate roots, or any identity drift.

Whole-state hashes are insufficient for long dossiers because they miss benchmark text embedded
inside otherwise novel context. R051 therefore requires exact raw and normalized field hashes,
substantial normalized-substring detection, sliding token-shingle/MinHash checks, upstream lineage
exclusion, and document/source IDs kept wholly within one split. All siblings and exhibits from a
dossier share a split; validation is also disjoint by generator family and policy template.

## Frozen internal gate

Arm B advances only if every condition holds on untouched internal panels:

| Measure | Gate |
|---|---:|
| Complete exact-action-set accuracy | `B - A >= 0.030`; semantic-cluster bootstrap 95% lower bound `> 0` |
| Long/dispersed-evidence exact actions | `B - A >= 0.050`; lower bound `> 0` |
| Separating-policy accuracy | `B - A >= 0.030` |
| Complete-outcome Brier | at least 5% relative improvement |
| Complete-outcome NLL | regression no greater than `0.02` |
| Overall field and each Choice/Noul/Score accuracy | each regression no greater than `0.01` |
| Both-order agreement | at least `0.99` |
| Mean paired probability delta | at most `0.02` |
| Any answer-order direction cell | regression no greater than `0.01` |
| English, natural Hebrew, and real-image accuracy | each regression no greater than `0.01` |
| Multimodal causal-use gap | correct-image versus shuffled/missing gap must not weaken |
| Protected replay Brier/NLL | relative regression no greater than 5% |
| Runtime | zero invalid outputs; p95 latency regression no greater than 10% at matched context |

Run one seed first. Only after the complete mechanism gate passes may two additional seeds run;
the exact-action gain must remain positive across three seeds with a cluster-bootstrap interval.

## Conditional RL experiment

Setwise RL becomes eligible only if the coverage treatment improves field accuracy and local
grouped outcomes but still fails a genuinely sequential, nondifferentiable workflow simulator.
The comparator is an exact supervised continuation trained on the same trajectories with matched
tokens and compute. The RL arm uses grouped trajectories, an enumerated-reward control variate
where possible, parent KL, and both exact-action and proper-calibration reward terms.

RL advances only if it adds at least two exact-outcome points over that matched comparator,
improves complete-outcome Brier by at least 5%, and passes the same Hebrew and multimodal gates.
Online policy gradients over a one-letter contextual-bandit response remain rejected because they
discard known full-distribution supervision.

## External gate remains unchanged

Only one frozen candidate may proceed to external evaluation:

- WorkflowEvals: security `>= .617`, trace `>= .716`, invoice `>= .618`, customer `>= .760`,
  equal-workflow mean `>= .678`, with all 705 cases and 1,524 decisions present;
- official JevBench v1.5.4: score `>= 72.13292897895896`, Intelligence
  `>= 71.99974990450261`, and 1,624/1,624 valid decisions;
- the same frozen Hebrew and multimodal preservation gates.

The 231-row public JevBench set remains diagnostic-only and cannot select this curriculum,
checkpoint, loss, temperature, or subsequent method.
