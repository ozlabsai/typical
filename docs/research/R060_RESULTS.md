# R060 results: GPT-labelled agent-trace data + KL-anchored continuation from R056

Date: 2026-10-22. Status log: `.context/r060/STATUS.md`. Raw outputs: `.context/r060/pod/`. Adapter: private `guychuk/typical-r060` (step 900 + `temperatures.json`, served id `r060-sha256-6bbe6b1567d7`).

## What changed vs R056
- **agent_trace targets:** gpt-6-astra where available, otherwise Qwen-only; Gemma was dropped. Astra covered 1,788 / 2,025 unique agent_trace questions (88%), labelled via OpenRouter until credit ran out, then via Codex until its usage limit (resets Oct 27).
- **Security choice/score fields:** astra coverage was only 19 / 1,292, so these stayed effectively as in R056.
- **Anchors** (`meta.anchor`, KL-to-base weight 1.0, base probabilities precomputed):
  - tasksource 7,000 q (commercial, decontaminated);
  - R055 HE/MX 5,500 q;
  - R056 EN replay 3,250 q.
- **Training:** continued from the R056 adapter, lr 3e-5 → 3e-6, 953 steps on 4×A100 (7.6 h). Selected step 900 on workflow-dev teacher-NLL + anchor-dev KL.

## Results

| | WF mean | security | agent trace | invoice | customer | EN 231 | HE / MX / native | ext. | contested | SQA real / null / AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| base (R056 serving path) | .550 | .392 | .550 | .547 | .711 | 202 | .852 / .835 / .815 | .790 | 55.0% | 85 / 99 / .850 |
| R056 | .649 | .604 | .635 | .627 | .729 | 196 | .822 / .835 / .866 | .803 | 59.4% | 114 / 124 / .981 |
| R056 λ=.5 | .601 | .471 | .577 | .609 | .748 | 202 | .852 / .843 / .899 | .809 | 58.3% | 110 / 117 / .972 |
| **R060** | **.648** | **.613** | **.644** | .600 | **.735** | 198 | .826 / .817 / .849 | .800 | 59.4% | **120 / 124 / .990** |
| Jev 1.13 | .678 | .617 | .716 | .618 | .760 | 200 | — | — | 62.6% | — |

The invoice primary-action score is .820 (R056 .787).

## Verdict
- **WorkflowEvals is flat (.648 vs .649).** Security, agent trace and customer each gain 0.6–0.9 points, and invoice loses 2.7.
- **The astra labels work per question but barely move the workflows.** Agent-trace choice/score argmax flips against consensus fell from .185 to .142, but the primary action only moved .635 → .644. Workflow decisions depend on several fields at once.
- **The KL anchors did not restore general skill.** EN 231 went 196 → 198 and Hebrew is flat. Continuing from R056 can only stop further drift; it cannot undo drift R056 already caused. Blending R056 to λ=.5 recovers more.
- **Images are the best of any Typical** (real 120/128, AUROC .990), and every retention gate passes.

## Lessons
- Run every pipeline as one on-pod chain. Two local network outages killed the orchestrating agents. The on-pod scripts kept running, but the final HF upload and pod stop were missed: the pod idled about 1 h after the eval finished, and the coordinator uploaded the adapter and stopped the pod by hand.
- The Codex CLI rate limit (until Oct 27) and the OpenRouter auto top-up lag both interrupted labelling. Budget for one paid backend with a high cap.
