import json, sys, statistics as st
sys.path.insert(0, "airdef")   # run from the repo root
import scorer
rows = [json.loads(l) for l in open(".context/airdef/situations.jsonl")]
m = scorer.load(sys.argv[1] if len(sys.argv) > 1 else ".context/typically/results/co_airdef_v4/best.pt")
key_of = lambda r, lab: [k for k in r["meta"]["costs"]][r["candidates"].index(lab)]
fams = {}
for r in rows:
    d = m.choice(r["state"], r["query"], r["candidates"]); d.pop("p_null")
    pick = key_of(r, max(d, key=d.get))
    c, best = r["meta"]["costs"], r["meta"]["best"]
    f = fams.setdefault(r["meta"]["family"], {"typical": [], "planner": [], "doctrine": []})
    f["typical"].append(c[pick] - c[best]); f["planner"].append(c[r["meta"]["planner"]] - c[best]); f["doctrine"].append(c[r["meta"]["doctrine"]] - c[best])
print(f"{'family':18s} {'n':>3s}  mean regret ($M-eq):  typical   planner   doctrine")
allr = {"typical": [], "planner": [], "doctrine": []}
for fam, v in fams.items():
    print(f"{fam:18s} {len(v['typical']):3d}  {'':20s} {st.mean(v['typical']):8.2f}  {st.mean(v['planner']):8.2f}  {st.mean(v['doctrine']):8.2f}")
    for k in allr: allr[k] += v[k]
print(f"{'all':18s} {len(allr['typical']):3d}  {'':20s} {st.mean(allr['typical']):8.2f}  {st.mean(allr['planner']):8.2f}  {st.mean(allr['doctrine']):8.2f}")
