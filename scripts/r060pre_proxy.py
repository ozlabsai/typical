#!/usr/bin/env python3
"""R060-pre: λ-scaled R056 adapters + the dev teacher-NLL proxy and its WorkflowEvals correlation.

  scale <src> <dst> <lam>                 copy adapter tree, every lora_B tensor × lam (= LoRA delta × lam)
  merge <src1> <w1> <src2> <w2> <dst>     rank-concatenated w1·delta1 + w2·delta2 (exact sum), r and alpha doubled
  sample <dev_wf.jsonl> <ids.txt>         ~2,000 dev questions, stratified by (workflow, question type);
                                          EN/HE twins drawn together
  score <name>=<devsel.json>... --out x   teacher-NLL / Brier / argmax agreement, per workflow and overall
  corr <proxy.json> <name>=<wf run dir>... --out x
                                          Spearman/Pearson of the proxy vs WF headline, per (model × workflow)
"""
import argparse
import json
import math
import random
import shutil
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

# WorkflowEvals headline metric per workflow (scripts/r053_calibration_headroom.HEADLINE) and devsel's id key
HEADLINE = {"security_incidents": "exact_actions", "agent_trace_observability": "primary_action",
            "invoice_processing": "exact_actions", "customer_service": "exact_actions"}
KEY = {"security": "security_incidents", "agent": "agent_trace_observability",
       "invoice": "invoice_processing", "customer": "customer_service"}


def scale(src, dst, lam):
    from safetensors import safe_open
    from safetensors.torch import save_file
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    with safe_open(src / "adapter_model.safetensors", "pt") as f:
        meta = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    hit = [k for k in tensors if "lora_B" in k]
    assert hit and all("lora_" in k for k in tensors), "unexpected adapter keys"
    for k in hit:
        tensors[k] = tensors[k] * lam  # dtype kept
    save_file(tensors, dst / "adapter_model.safetensors", metadata=meta)
    for name in ("adapter_config.json", "temperatures.json"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    print(f"{dst}: {len(hit)} lora_B tensors x {lam}")


def merge(src1, w1, src2, w2, dst):
    """Rank concatenation, the exact weighted sum of the two LoRA deltas: A = [A1; A2], B = [w1·B1, w2·B2],
    r = r1 + r2 and alpha scaled so alpha/r (the per-delta scaling) is unchanged. Temperatures from src1."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    src1, src2, dst = Path(src1), Path(src2), Path(dst)
    c1, c2 = (json.loads((s / "adapter_config.json").read_text()) for s in (src1, src2))
    same = ("r", "lora_alpha", "target_modules", "use_rslora", "use_dora", "rank_pattern", "alpha_pattern")
    assert all(c1.get(k) == c2.get(k) for k in same) and not c1.get("rank_pattern"), "incompatible adapters"
    t1, t2 = ({k: f.get_tensor(k) for k in f.keys()} for f in (safe_open(s / "adapter_model.safetensors", "pt")
                                                              for s in (src1, src2)))
    assert t1.keys() == t2.keys()
    out = {}
    for k in t1:
        if "lora_A" in k:
            out[k] = torch.cat([t1[k], t2[k]], 0)
        else:
            assert "lora_B" in k, k
            out[k] = torch.cat([w1 * t1[k], w2 * t2[k]], 1)
    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, dst / "adapter_model.safetensors", metadata={"format": "pt"})
    c1["r"], c1["lora_alpha"] = 2 * c1["r"], 2 * c1["lora_alpha"]
    (dst / "adapter_config.json").write_text(json.dumps(c1, indent=2) + "\n")
    if (src1 / "temperatures.json").exists():
        shutil.copy(src1 / "temperatures.json", dst / "temperatures.json")
    print(f"{dst}: {w1}·{src1.name} + {w2}·{src2.name}, r={c1['r']} alpha={c1['lora_alpha']}")


def qtype(example):
    labels = [k for k, _ in example["items"]]
    return "noul" if set(labels) == {"true", "false"} else "score" if all(l.isdigit() for l in labels) else "choice"


def sample(dev, out, n=2000, seed=60):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.train_gemma_decision_lora import load_examples
    ex = load_examples([dev])
    units = defaultdict(lambda: defaultdict(list))  # stratum -> unit (EN/HE twin key) -> ids
    for e in ex:
        units[(e["id"].split("-")[1], qtype(e))][e.get("twin") or e["id"].split(":", 1)[1]].append(e["id"])
    frac, ids = n / len(ex), []
    for stratum in sorted(units):
        keys = sorted(units[stratum])
        random.Random(f"{seed}-{stratum}").shuffle(keys)
        for k in keys[:round(frac * len(keys))]:
            ids += units[stratum][k]
    Path(out).write_text("\n".join(sorted(ids)) + "\n")
    print(f"{len(ids)} of {len(ex)} questions, {len(units)} strata")


def metrics(rows):
    nll = [-sum(t * math.log(max(s, 1e-12)) for t, s in zip(r["teacher"], r["student"])) for r in rows]
    brier = [sum((s - t) ** 2 for t, s in zip(r["teacher"], r["student"])) for r in rows]
    am = lambda v: max(range(len(v)), key=v.__getitem__)  # noqa: E731
    agree = [am(r["teacher"]) == am(r["student"]) for r in rows]
    return {"nll": st.mean(nll), "brier": st.mean(brier), "agree": st.mean(agree), "n": len(rows)}


def score(pairs, out):
    report = {}
    for pair in pairs:
        name, path = pair.split("=", 1)
        rows = json.loads(Path(path).read_text())["rows"]
        by = defaultdict(list)
        for r in rows:
            by["all"].append(r)
            by[KEY[r["id"].split("-")[1]]].append(r)
            by["he" if r["id"].split("/")[0].endswith("-he") else "en"].append(r)  # HE twins: <req>-he/<qid>
        report[name] = {k: metrics(v) for k, v in by.items() if v}
    Path(out).write_text(json.dumps(report, indent=1) + "\n")
    cols = ["all", *KEY.values(), "en", "he"]
    print("| model | " + " | ".join(cols) + " |  (teacher-NLL / Brier / argmax agreement)")
    for name, m in report.items():
        print(f"| {name} | " + " | ".join(
            f"{m[c]['nll']:.3f} / {m[c]['brier']:.3f} / {m[c]['agree']:.3f}" if c in m else "-" for c in cols) + " |")


def wf_headline(run_dir):
    out = {}
    for wf, metric in HEADLINE.items():
        f = Path(run_dir) / wf / "scores.json"
        if f.exists():
            s = json.loads(f.read_text())
            out[wf] = next(x["metrics"] for x in s["series"] if x["label_set_id"] == "labels-consensus")[metric]
    return out


def corr(proxy_path, pairs, out):
    proxy = json.loads(Path(proxy_path).read_text())
    pts = []  # (model, workflow, wf score, nll, brier, agree)
    for pair in pairs:
        name, run = pair.split("=", 1)
        for wf, v in wf_headline(run).items():
            if name in proxy and wf in proxy[name]:
                p = proxy[name][wf]
                pts.append((name, wf, v, p["nll"], p["brier"], p["agree"]))

    def ranks(v):  # average ranks for ties (Spearman = Pearson on ranks)
        order = sorted(range(len(v)), key=v.__getitem__)
        out, i = [0.0] * len(v), 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out

    def rho(xs, ys):
        if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
            return {"spearman": None, "pearson": None, "n": len(xs)}
        return {"spearman": st.correlation(ranks(xs), ranks(ys)), "pearson": st.correlation(xs, ys), "n": len(xs)}

    def block(sel):
        wf = [p[2] for p in sel]
        return {"-nll": rho(wf, [-p[3] for p in sel]), "-brier": rho(wf, [-p[4] for p in sel]),
                "agree": rho(wf, [p[5] for p in sel])}

    def centered(sel):  # subtract each workflow's mean from both sides: only the model ranking within a workflow
        mean = {w: [st.mean(p[i] for p in sel if p[1] == w) for i in (2, 3, 4, 5)] for w in {p[1] for p in sel}}
        return [(p[0], p[1], *(p[i] - mean[p[1]][i - 2] for i in (2, 3, 4, 5))) for p in sel]

    models = sorted({p[0] for p in pts})
    report = {"points": pts, "pooled": block(pts), "pooled_workflow_centered": block(centered(pts)),
              "within_workflow": {w: block([p for p in pts if p[1] == w]) for w in HEADLINE},
              "model_mean": block([(m, "mean", *(st.mean(p[i] for p in pts if p[0] == m) for i in (2, 3, 4, 5)))
                                   for m in models if sum(p[0] == m for p in pts) == 4])}
    Path(out).write_text(json.dumps(report, indent=1) + "\n")
    fmt = lambda r: "-" if r["spearman"] is None else f"{r['spearman']:+.2f} / {r['pearson']:+.2f} (n={r['n']})"  # noqa: E731
    print("| set | -NLL | -Brier | agreement |   (Spearman / Pearson vs WF headline)")
    for k in ("pooled", "pooled_workflow_centered", "model_mean"):
        print(f"| {k} | " + " | ".join(fmt(report[k][m]) for m in ("-nll", "-brier", "agree")) + " |")
    for w, b in report["within_workflow"].items():
        print(f"| {w} | " + " | ".join(fmt(b[m]) for m in ("-nll", "-brier", "agree")) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scale"); p.add_argument("src"); p.add_argument("dst"); p.add_argument("lam", type=float)
    p = sub.add_parser("merge"); p.add_argument("src1"); p.add_argument("w1", type=float); p.add_argument("src2")
    p.add_argument("w2", type=float); p.add_argument("dst")
    p = sub.add_parser("sample"); p.add_argument("dev"); p.add_argument("out"); p.add_argument("--n", type=int, default=2000)
    p = sub.add_parser("score"); p.add_argument("pairs", nargs="+"); p.add_argument("--out", required=True)
    p = sub.add_parser("corr"); p.add_argument("proxy"); p.add_argument("pairs", nargs="+"); p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "scale":
        scale(a.src, a.dst, a.lam)
    elif a.cmd == "merge":
        merge(a.src1, a.w1, a.src2, a.w2, a.dst)
    elif a.cmd == "sample":
        sample(a.dev, a.out, a.n)
    elif a.cmd == "score":
        score(a.pairs, a.out)
    else:
        corr(a.proxy, a.pairs, a.out)


if __name__ == "__main__":
    main()
