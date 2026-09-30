"""TypeSafe WorkflowEvals -> our eval_wf jsonl format.

Four public datasets (typesafe/evalsafe-*) at evals.typesafe.ai, decision-level:
each row is one question instance with {type, instructions, criteria}, the case state,
and a consensus answer. That is our interface exactly -- query_text() already consumes
this shape -- so no re-rendering is needed.

Gold is a CONSENSUS of gpt-6-astra + claude-opus-5, not ground truth. Accuracy here is
agreement with two frontier models, which is what the published leaderboard measures too.

uv run scripts/workflowevals.py [--out data_wfe/eval]
"""
import argparse, json, pathlib, sys
import pandas as pd
from huggingface_hub import hf_hub_download
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from jev_hf_datasets import row  # same row/query_text contract as every other eval file

SETS = ["invoice-processing", "customer-service", "security-incidents", "agent-trace-observability"]

def build(name):
    p = hf_hub_download(f"typesafe/evalsafe-{name}", "data/questions.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    out, skipped = [], {"unanswered": 0, "null_answer": 0, "answer_not_in_options": 0}
    for _, r in df.iterrows():
        q = json.loads(r.question_json)
        crit = q.get("criteria")
        cons = r.consensus if isinstance(r.consensus, dict) else json.loads(r.consensus)
        if cons.get("status") != "answered":
            skipped["unanswered"] += 1; continue
        ans = cons.get("answer_json")
        if isinstance(ans, str):
            try: ans = json.loads(ans)
            except Exception: pass
        # three shapes in the suite: noul with no criteria (plain true/false), dict criteria
        # (option -> description; options are the keys), and score with an ordered list of
        # level descriptions (answers are the level indices as strings).
        if isinstance(crit, dict):   cands = list(crit.keys())
        elif isinstance(crit, list): cands = [str(i) for i in range(len(crit))]
        else:                        cands = ["false", "true"]
        ans_s = str(ans).lower() if isinstance(ans, bool) else str(ans)
        if ans is None or ans_s in ("null", "none") and ans_s not in cands:
            skipped["null_answer"] += 1; continue
        if ans_s not in cands:
            skipped["answer_not_in_options"] += 1; continue
        out.append(row(f"wfe_{name.replace('-','_')}", json.loads(r.state_json), q, cands,
                       label=cands.index(ans_s),
                       meta={"family": name, "kind": r.kind, "case_id": r.case_id,
                             "node_id": r.node_id, "question_id": r.question_id,
                             "consensus_confidence": cons.get("confidence")}))
    return out, skipped

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data_wfe/eval")
    a = ap.parse_args()
    d = pathlib.Path(a.out); d.mkdir(parents=True, exist_ok=True)
    grand = 0
    for s in SETS:
        rows, sk = build(s)
        f = d / f"wfe_{s.replace('-','_')}.jsonl"
        with f.open("w") as fh:
            for r in rows: fh.write(json.dumps(r) + "\n")
        from collections import Counter
        print(f"{f.name:40s} {len(rows):5d} rows  kinds={dict(Counter(r['meta']['kind'] for r in rows))}  skipped={ {k:v for k,v in sk.items() if v} }")
        grand += len(rows)
    print(f"{'TOTAL':40s} {grand:5d}")
