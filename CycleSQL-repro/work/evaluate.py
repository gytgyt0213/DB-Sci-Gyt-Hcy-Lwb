"""Evaluate the CycleSQL run and the top-1 baseline with the paper's own scorer.

`spider_utils.eval.evaluation` needs a gold file whose entries line up 1:1 with
the prediction file, so we rebuild the wta_1 subset of dev.json in exactly the
order `select_indices()` produced predictions in.

Usage:  python work/evaluate.py [db_id]
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.join(ROOT, "src_repo", "CycleSQL-main")
DATA = os.path.join(ROOT, "data", "spider")
DB_ID = sys.argv[1] if len(sys.argv) > 1 else "wta_1"
OUT = os.path.join(ROOT, "outputs", "spider", "resdsql")
LOGS = os.path.join(ROOT, "work", "logs")
os.makedirs(LOGS, exist_ok=True)

dev = json.load(open(os.path.join(DATA, "dev.json"), encoding="utf-8"))
gold_subset = [ex for ex in dev if ex["db_id"] == DB_ID]
gold_path = os.path.join(DATA, f"gold_{DB_ID}.json")
with open(gold_path, "w", encoding="utf-8") as f:
    json.dump(gold_subset, f, ensure_ascii=False, indent=4)
print(f"gold subset: {len(gold_subset)} questions over '{DB_ID}' -> {gold_path}")


def run_eval(pred_file, tag):
    cmd = [sys.executable, "-m", "spider_utils.eval.evaluation",
           "--gold", gold_path, "--pred", pred_file, "--db", os.path.join(DATA, "database")]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    log_path = os.path.join(LOGS, f"eval_{tag}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout + "\n" + proc.stderr)
    nums = re.findall(r"([-+]?\d*\.\d+|\d+)", proc.stdout)
    return proc.returncode, log_path, proc.stdout


results = {}
for tag, pred in (("baseline_top1", os.path.join(OUT, "preds_baseline_top1.txt")),
                  ("cyclesql", os.path.join(OUT, "preds.txt"))):
    if not os.path.exists(pred):
        print(f"!! missing {pred}")
        continue
    rc, log_path, out = run_eval(pred, tag)
    print("=" * 90)
    print(f"{tag}  (rc={rc})   log -> {log_path}")
    print("=" * 90)
    print(out.strip()[-2500:])
    results[tag] = {"log": log_path, "returncode": rc}

with open(os.path.join(OUT, "eval_summary.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
