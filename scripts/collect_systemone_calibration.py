"""Collect internal SystemOne probabilities for calibration fitting."""

import argparse
import json
import random
import sys
import urllib.request


def records_for(row, response):
    records = []
    for qid, question in row["questions"].items():
        answer = response["answers"][qid]
        if question["type"] == "score":
            labels = [str(i) for i in range(len(question["criteria"]))]
        else:
            labels = [str(label) for label in question["criteria"]]
        soft = row.get("soft_targets", {}).get(qid)
        target = ({label: soft[i] for i, label in enumerate(labels)} if soft is not None
                  else str(row["targets"][qid]["label"]))
        if question["type"] == "noul":
            yes = float(answer["noul"])
            probabilities = {"true": yes, "false": 1.0 - yes}
        else:
            probabilities = answer["probabilities"]
        records.append({"id": f"{row.get('id')}/{qid}", "qtype": question["type"],
                        "target": target, "probabilities": probabilities})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with open(args.input) as source:
        rows = [json.loads(line) for line in source if line.strip()]
    random.Random(args.seed).shuffle(rows)
    for index, row in enumerate(rows[:args.limit], 1):
        payload = json.dumps({"model": args.model, "state": row["state"],
                              "questions": row["questions"]}).encode()
        request = urllib.request.Request(args.endpoint.rstrip("/") + "/v1/systemone", data=payload,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            answer = json.loads(response.read())
        for record in records_for(row, answer):
            print(json.dumps(record, ensure_ascii=False), flush=True)
        if index % 100 == 0:
            print(f"{index}/{min(args.limit, len(rows))}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
