"""build_import + build_from_plan (scripts/typically_spike.py): schema, splits, mapping, balancing, flips, policy, soft targets, LLM enrichment (fake client). No model."""
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

from scripts.typically_spike import build_import

RECORDS = [{"t": f"ticket {i}", "team": "A" if i % 2 else "B", "esc": "Yes" if i % 10 == 0 else "no", "urg": str(i % 4)}
           for i in range(100)]
DECISIONS = [{"column": "team", "question": "Which team?", "type": "choice"},
             {"column": "esc", "question": "Escalate?", "type": "noul"},
             {"column": "urg", "question": "How urgent?", "type": "score"}]


def test_build_import():
    s = build_import(RECORDS, "t", DECISIONS, random.Random(0), balance=False, flips=0)
    assert set(s) == {"train", "val", "import_oneliner"}
    assert {k: len(v) // 3 for k, v in s.items()} == {"train": 70, "val": 10, "import_oneliner": 20}
    r = next(r for r in s["train"] if r["task"] == "import_esc")
    assert set(r) == {"state", "query", "candidates", "target", "p_null", "task", "label", "meta"}
    assert r["candidates"] == ["no", "yes"] and r["target"][r["label"]] == 1.0
    assert {x["state"] for v in s.values() for x in v} == {x["t"] for x in RECORDS}
    assert len({x["state"] for k in ("train", "val") for x in s[k]} & {x["state"] for x in s["import_oneliner"]}) == 0

    b = build_import(RECORDS, "t", DECISIONS, random.Random(0), flips=0)
    c = Counter(x["label"] for x in b["train"] if x["task"] == "import_esc")
    assert c[1] >= c[0] // 3 - 1 and c[1] > Counter(x["label"] for x in s["train"] if x["task"] == "import_esc")[1]
    assert len(b["val"]) == len(s["val"]) and len(b["import_oneliner"]) == len(s["import_oneliner"])

    with pytest.raises(ValueError):
        build_import([{"t": "x", "e": "maybe"}], "t", [{"column": "e", "question": "?", "type": "noul"}], random.Random(0))


def test_import_flips():
    words = ["alpha", "bravo", "cargo", "delta", "eagle"]   # each in 20% of the cases
    recs = [{**r, "t": f"ticket {i} {words[i % 5]}"} for i, r in enumerate(RECORDS)]
    build = lambda **kw: build_import(recs, "t", DECISIONS, random.Random(0), balance=False, **kw)
    s, off = build(), build(flips=0)
    held = lambda sp: {k: sorted(json.dumps(x) for x in sp[k]) for k in ("val", "import_oneliner")}
    assert held(s) == held(off)   # no flip or ms_group row outside train
    assert not any("rflip" in x["task"] or "ms_group" in x["meta"] for k in ("val", "import_oneliner") for x in s[k])
    assert not any("rflip" in x["task"] or "ms_group" in x["meta"] for x in off["train"])
    flips = [x for x in s["train"] if x["task"] == "import_rflip"]
    assert 0.03 < len(flips) / (70 * 3) < 0.20   # ~10% of train (case, decision) pairs
    plain = {x["meta"]["ms_group"]: x for x in s["train"] if x["task"] != "import_rflip" and "ms_group" in x["meta"]}
    for f in flips:
        p = plain[f["meta"]["ms_group"]]
        assert p["state"] == f["state"] and p["candidates"] == f["candidates"] and f["query"].startswith(p["query"])
        word = re.search(r'mentions "(\w+)"', f["query"]).group(1)
        hit = word in f["state"].split()
        if p["candidates"] == ["no", "yes"] and "answer yes only" in f["query"]:
            assert f["candidates"][f["label"]] == ("yes" if hit else "no")
        else:
            x, y = re.search(r"pick (.+) when .*, otherwise (.+)\.$", f["query"]).groups()
            assert f["candidates"][f["label"]] == (x if hit else y)


# ---------------------------------------------------------------- build_from_plan (scripts/typically_spike.py)
import copy
from types import SimpleNamespace

from scripts import typically_plan as tp
from scripts import typically_spike as ts
from scripts.typically_spike import build_from_plan

ENRICH = {"balance": True, "dedupe_soft": True, "policy": {}, "synthetic": False, "languages": []}
SETTINGS = {"holdout": 20, "seed": 0}


def company(n=300):
    """plan = a fact column; esc = Yes/Y/TRUE/no/N variants; sev = low < medium < high (alphabetical would be high, low, medium),
    high is rare; every 10th case is repeated with the opposite esc answer (conflicting duplicates)."""
    recs = [{"plan": ["starter", "pro", "enterprise"][i % 3], "msg": f"Customer message number {i} about a shipment problem in detail {i * 7}",
             "esc": ["Yes", "Y", "TRUE", "no", "N", "No"][i % 6], "sev": "high" if i % 40 == 7 else ["low", "medium"][i % 2]} for i in range(n)]
    return recs + [{**r, "esc": "no" if r["esc"] in ("Yes", "Y", "TRUE") else "Yes"} for r in recs[::10]]


def company_plan(recs):
    plan = tp.heuristic_plan(tp.profile(recs), recs)
    plan["decisions"] = [d for d in plan["decisions"] if d["column"] != "plan"]
    plan["case"]["parts"].append({"column": "plan", "role": "fact", "sentence": "The customer is on the {value} plan.", "confidence": 1, "reason": ""})
    return plan


def build(recs, plan, llm=None, **enrich):
    return build_from_plan(recs, plan, {**ENRICH, **enrich}, SETTINGS, random.Random(0), llm)


def test_build_from_plan_sample_file():
    recs = tp.load_records({"kind": "sample"})
    plan = tp.heuristic_plan(tp.profile(recs), recs)
    split, st = build(recs, plan)
    assert set(split) == {"train", "val", "import_oneliner"} and st["rows"] == {k: len(v) for k, v in split.items()}
    assert {r["candidates"][0] for r in split["train"] if r["task"] == "import_urgency"} == {"0"}
    urg = next(r for r in split["train"] if r["task"] == "import_urgency")
    assert urg["candidates"] == ["0", "1", "2", "3"] and urg["query"] == "What urgency level?"
    assert not {r["state"] for r in split["train"] + split["val"]} & {r["state"] for r in split["import_oneliner"]}   # duplicates never straddle
    assert st["warnings"] == [] and st["flips"] > 0 and st["labels"]["before"]["team"] and st["labels"]["after"]["team"]


def test_build_from_plan_order_mapping_split_flips_policy():
    recs = company()
    plan = company_plan(recs)
    dec = {d["column"]: d for d in plan["decisions"]}
    assert dec["sev"]["labels"] == ["low", "medium", "high"] and dec["esc"]["mapping"]["y"] == "yes"
    split, st = build(recs, plan, policy={"sev": "High means an outage."})
    by = lambda name, task: [r for r in split[name] if r["task"] == task]
    assert {tuple(r["candidates"]) for r in by("train", "import_sev")} == {("low", "medium", "high")}   # plan order, never sorted
    assert {tuple(r["candidates"]) for r in by("import_oneliner", "import_esc")} == {("no", "yes")}
    assert st["labels"]["before"]["esc"]["yes"] > 0   # Yes / Y / TRUE all map to yes
    states = lambda *names: {r["state"] for n in names for r in split[n]}
    assert not states("train", "val") & states("import_oneliner") and all(r["state"].startswith("The customer is on the ") for r in split["train"])
    # flips: ~10% of the base train rows, on the fact column, paired through ms_group with a plain row on the same case
    flips = [r for r in split["train"] if r["task"] == "import_rflip"]
    base = [r for r in split["train"] if r["task"] != "import_rflip"]
    assert 0.05 < len(flips) / len({id(r) for r in base}) < 0.16 and st["flips"] == len(flips)
    plain = {r["meta"]["ms_group"]: r for r in base if "ms_group" in r["meta"]}
    for f in flips:
        p = plain[f["meta"]["ms_group"]]
        assert p["state"] == f["state"] and f["query"].startswith(p["query"])
        plan_of = re.search(r"on the (\w+) plan", f["state"]).group(1)
        word = re.search(r"when the case says plan is (\w+)", f["query"]).group(1)
        if "answer yes only" in f["query"]:
            assert f["candidates"][f["label"]] == ("yes" if plan_of == word else "no")
        else:
            x, y = re.search(r"pick (.+) when .*, otherwise (.+)\.$", f["query"]).groups()
            assert f["candidates"][f["label"]] == (x if plan_of == word else y)
    # policy: ~30% of the sev train rows carry the text (same label), no other decision does
    sev = by("train", "import_sev")
    with_policy = [r for r in sev if r["query"].endswith("\nHigh means an outage.")]
    assert r"What sev level?" in sev[0]["query"] and 0.15 < len(with_policy) / len(sev) < 0.45
    assert not any("outage" in r["query"] for r in split["train"] if r["task"] == "import_esc")
    assert not any("outage" in r["query"] for n in ("val", "import_oneliner") for r in split[n])


def test_build_from_plan_null_mapping_and_soft_targets():
    recs = company()
    plan = company_plan(recs)
    next(d for d in plan["decisions"] if d["column"] == "esc")["mapping"]["y"] = None   # "Y" = no answer: those rows are dropped
    _, st = build(recs, plan)
    assert st["dropped"]["esc"] == sum(r["esc"] == "Y" for r in recs)
    plan = company_plan(recs)
    split, st = build(recs, plan)
    soft = [r for r in split["train"] if r["meta"].get("soft")]
    assert soft and st["soft_rows"] == len(soft)   # the repeated cases with opposite answers merge
    for r in soft:
        assert abs(sum(r["target"]) - 1) < 1e-9 and 0 < max(r["target"]) < 1 and r["label"] == r["target"].index(max(r["target"]))
    hard, st = build(recs, plan, dedupe_soft=False)
    assert st["soft_rows"] == 0 and all(sum(r["target"]) == 1 and max(r["target"]) == 1 for r in hard["train"])


def test_build_from_plan_llm_flags_without_key_are_ignored():
    recs = company()
    split, st = build(recs, company_plan(recs), synthetic=True, languages=["es"])
    assert set(split) == {"train", "val", "import_oneliner"} and st["synthetic"] == 0 and "not available" in st["warnings"][0]
    assert not any(r["meta"].get("synthetic") or "lang" in r["meta"] for r in split["train"])


class FakeLLM:
    """Stands in for typically_llm.complete_json: translates by prefixing [lang], writes numbered synthetic cases."""
    def __init__(self, fixed=None, usage=None):
        self.calls, self.max_tokens, self.fixed, self.usage = [], [], fixed, usage   # fixed: every returned text; usage: (in, out) tokens per call

    def __call__(self, system, prompt, schema, *, max_tokens):
        self.calls.append(prompt)
        self.max_tokens.append(max_tokens)
        assert schema is ts.TEXTS
        if prompt.startswith("Translate"):
            lang = re.search(r"code '(\w+)'", prompt).group(1)
            texts = [f"[{lang}] {s}" for s in json.loads(prompt.split("\n\n", 1)[1])]
        else:
            n = int(re.search(r"Write (\d+) NEW", prompt).group(1))
            texts = [f"synthetic case {len(self.calls)}-{k} in the style of the real ones" for k in range(n)]
        texts = [self.fixed] * len(texts) if self.fixed else texts
        i, o = self.usage or (len(prompt) // 4, len(json.dumps(texts)) // 4)
        return {"texts": texts}, {"input_tokens": i, "output_tokens": o}


def test_build_from_plan_llm_enrichment_with_fake_client():
    recs = company()
    plan = company_plan(recs)
    plain, _ = build(recs, plan)
    llm = FakeLLM()
    split, st = build(recs, plan, llm, synthetic=True, languages=["es"])
    assert st["warnings"] == [] and 0 < st["llm_cost_usd"] < 2 and llm.calls
    canon = lambda rows: sorted(json.dumps(r, sort_keys=True) for r in rows)
    assert canon(split["val"]) == canon(plain["val"]) and canon(split["import_oneliner"]) == canon(plain["import_oneliner"])   # never enriched
    syn = [r for r in split["train"] if r["meta"].get("synthetic")]
    assert syn and st["synthetic"] == len(syn) <= 0.15 * len(plain["train"])
    sev = Counter(r["candidates"][r["label"]] for r in syn if r["task"] == "import_sev")
    real = st["labels"]["before"]["sev"]
    assert sev["high"] == sev["medium"] + sev["low"] and 0 < sev["high"] <= real["high"]   # capped at 1x real; equal number for the majority
    es = [r for r in split["train"] if r["meta"].get("lang") == "es"]
    assert es and all(r["state"].startswith("[es] ") and "ms_group" not in r["meta"] for r in es)
    assert st["translated"]["train"] == len(es) and 0.1 < len({r["state"] for r in es}) / len({r["state"] for r in plain["train"]}) < 0.3
    ev = split["import_es"]
    assert ev and st["translated"]["eval"] == {"es": len(ev)} and all(r["state"].startswith("[es] ") for r in ev)
    orig = {r["state"] for r in split["import_oneliner"]}
    assert all(r["state"][5:] in orig for r in ev) and not {r["state"] for r in ev} & {r["state"] for r in split["train"]}


def test_soft_dedup_groups_by_normalised_state_and_score_stays_hard():
    choice = {"column": "c", "question": "q?", "labels": ["a", "b"], "type": "choice"}
    (r,) = ts.plan_rows([("Hello  World!", "a"), ("hello world", "b"), ("HELLO-world", "b")], choice, True)
    assert r["state"] == "Hello  World!" and r["meta"]["soft"] and r["target"] == pytest.approx([1 / 3, 2 / 3]) and r["label"] == 1
    score = {**choice, "labels": ["0", "1", "2"], "type": "score"}   # train.py --ordinal_smooth would overwrite a soft target
    (r,) = ts.plan_rows([("x", "0"), ("X", "2"), ("x.", "2")], score, True)
    assert r["candidates"][r["label"]] == "2" and "soft" not in r["meta"] and r["target"] == [0, 0, 1]
    assert len(ts.plan_rows([("x", "0"), ("X", "0")], score, False)) == 2


def test_llm_enrichment_drops_normalised_collisions_with_the_other_side():
    recs = company()
    plan = company_plan(recs)
    plain, _ = build(recs, plan)
    val = plain["val"][0]["state"]
    split, st = build(recs, plan, FakeLLM(fixed=val.upper() + "!"), languages=["es"])   # every translation is the val case, respelled
    norm_of = lambda rows: {tp.norm(r["state"]) for r in rows}
    assert tp.norm(val) not in norm_of(split["train"]) and st["collisions"] > 0 and st["translated"]["train"] == 0
    train = plain["train"][0]["state"]
    split, st = build(recs, plan, FakeLLM(fixed=train.upper() + "!"), languages=["es"])   # every translation is a train case, respelled
    assert split["import_es"] == [] and st["translated"]["eval"] == {"es": 0} and st["collisions"] > 0
    assert not norm_of(split["import_oneliner"]) & norm_of(split["train"])


def test_llm_spend_is_charged_from_usage():
    recs = company()
    plan = company_plan(recs)
    llm = FakeLLM(usage=(1000, 2000))   # $0.044 a call
    _, st = build(recs, plan, llm, synthetic=True, languages=["es", "de"])
    assert len(llm.calls) > 3 and st["warnings"] == [] and st["llm_cost_usd"] == round(0.044 * len(llm.calls), 2)
    assert all(m < 16000 for m in llm.max_tokens)   # sized to the call, not the old 16000


def test_build_from_plan_reports_enrichment_progress():
    recs, seen = company(), []
    build_from_plan(recs, company_plan(recs), {**ENRICH, "synthetic": True, "languages": ["es"]}, SETTINGS, random.Random(0), FakeLLM(),
                    lambda *a: seen.append(a))
    assert {k for k, _, _ in seen} == {"synthetic", "translate_es"}
    for kind in ("synthetic", "translate_es"):
        done = [(d, t) for k, d, t in seen if k == kind]
        assert [d for d, _ in done] == sorted(d for d, _ in done) and done[-1][0] == done[-1][1] > 0   # counts up to its total, in cases


def test_build_from_plan_bad_language():
    llm = FakeLLM()
    with pytest.raises(ValueError, match="language"):
        build(company(), company_plan(company()), llm, languages=["../x"])


def test_build_endpoint_v2(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "site"))   # as tests/test_typically_job.py
    import server
    import typically_analyze
    from fastapi.testclient import TestClient
    recs = company()
    monkeypatch.setattr(server, "TYPICALLY", tmp_path)
    monkeypatch.setattr(typically_analyze, "_records", {"a" * 32: recs})
    body = {"records_token": "a" * 32, "plan": company_plan(recs), "name": "Acme Co", "base": "medium", "enrich": {"policy": {"sev": "High means an outage."}},
            "settings": {"steps": 200, "holdout": 25, "seed": 3}}
    c = TestClient(server.app)
    r = c.post("/api/typically/build", json=body)
    assert r.status_code == 200, r.text
    out, res = tmp_path / "jobs" / "acme_co", r.json()
    assert res["job"] == "acme_co" and res["stats"]["rows"] == res["splits"] and "--steps 200" in res["command"] and "Qwen3.5-4B" in res["command"]
    assert len((out / "train.jsonl").read_text().splitlines()) == res["splits"]["train"] and (out / "eval" / "import_oneliner.jsonl").exists()
    assert json.loads((out / "job.json").read_text()) == {"name": "Acme Co", "base": "medium", "steps": 200, "records_token": "a" * 32}
    assert json.loads((out / "settings.json").read_text())["holdout"] == 25 and json.loads((out / "enrich.json").read_text())["policy"]
    assert json.loads((out / "plan.json").read_text())["decisions"]
    bad = copy.deepcopy(body)
    bad["plan"]["decisions"][0]["mapping"]["nope"] = "yes"
    assert c.post("/api/typically/build", json=bad).status_code == 400
    assert c.post("/api/typically/build", json={**body, "records_token": "b" * 32}).status_code == 404


def test_reveal_baseline_and_catch():
    from scripts.typically_reveal import reveal
    noul = lambda i, gold: {"state": f"case {i}", "query": "Reply?", "candidates": ["no", "yes"], "label": gold, "task": "import_reply",
                            "meta": {"qtype": "noul"}}
    rows = [noul(i, int(i in (0, 3))) for i in range(10)] + [
        {"state": f"case {i}", "query": "Folder?", "candidates": ["a", "b"], "label": g, "task": "import_folder", "meta": {"qtype": "choice"}}
        for i, g in ((0, 0), (1, 0), (2, 1))]
    std = {0: .9, 1: .8}   # p(yes): top 20% = cases 0, 1 -> catches 1 of the 2 yes
    yours = {3: .7, 0: .5, 5: .5}   # cases 3, 0 (0 before the tied 5: stable) -> catches both
    res = lambda p, i, q: ({"probs": {"no": 1 - p.get(i, .1), "yes": p.get(i, .1)}, "p_null": 0, "argmax": "yes" if p.get(i, .1) > .5 else "no"}
                           if q == "Reply?" else {"probs": {"a": .6, "b": .4}, "p_null": 0, "argmax": "a"})
    run = lambda state, group: [[res(p, int(state.split()[1]), r["query"]) for r in group] for p in (std, yours)]
    d = {x["key"]: x for x in reveal(rows, run)["decisions"]}
    assert d["reply"]["majority"] == "no" and d["reply"]["baseline"] == 0.8 and d["reply"]["catch"] == {"standard": 0.5, "yours": 1.0}
    assert d["folder"]["majority"] == "a" and d["folder"]["baseline"] == 2 / 3 and "catch" not in d["folder"]


def test_teach_endpoint_prepares_the_examples_inside_the_job(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "site"))
    import server
    import typically_analyze
    import typically_job as tj
    from fastapi.testclient import TestClient
    recs = company()
    for m in (server, tj):
        monkeypatch.setattr(m, "TYPICALLY", tmp_path)
    monkeypatch.setattr(typically_analyze, "_records", {"a" * 32: recs})
    ran = []
    monkeypatch.setattr(tj, "run", lambda slug, log, base, steps, prepared=False: ran.append((slug, base, steps, prepared)))
    body = {"records_token": "a" * 32, "plan": company_plan(recs), "name": "Acme Co", "base": "medium", "settings": {"steps": 200}}
    c = TestClient(server.app)

    def settle():
        for _ in range(200):
            if not server._jobs:
                return
            time.sleep(0.02)
    r = c.post("/api/typically/teach", json=body)
    assert r.status_code == 200 and r.json()["job"] == "acme_co" and r.json()["code"] == "preparing"
    settle()
    st = tj.read_status("acme_co")
    rows = len((tmp_path / "jobs/acme_co/train.jsonl").read_text().splitlines())
    assert ran == [("acme_co", "medium", 200, True)] and st["events"][0]["key"] == "prepare" and st["events"][0]["n"] == rows
    assert json.loads((tmp_path / "jobs/acme_co/job.json").read_text())["name"] == "Acme Co"
    bad = copy.deepcopy(body)
    bad["plan"]["decisions"][0]["mapping"]["nope"] = "yes"
    assert c.post("/api/typically/teach", json=bad).status_code == 400   # the fast checks answer the request itself
    monkeypatch.setattr(server, "build_from_plan", lambda *a: (_ for _ in ()).throw(ValueError("no rows left")))
    assert c.post("/api/typically/teach", json=body).status_code == 200
    settle()
    st = tj.read_status("acme_co")
    assert st["phase"] == "failed" and st["code"] == "failed_build" and "no rows left" in st["message"] and len(ran) == 1
