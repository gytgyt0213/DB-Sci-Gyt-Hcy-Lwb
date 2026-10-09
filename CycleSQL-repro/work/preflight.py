"""One-shot preflight check for the CycleSQL reproduction.

Run this BEFORE the real run. It checks (and where possible fixes) every
external dependency the pipeline needs, then performs a 1-question end-to-end
smoke test so that runtime errors surface here rather than halfway through a
long run.

    python work/preflight.py
"""
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # CycleSQL-repro/
REPO = os.path.join(ROOT, "src_repo", "CycleSQL-main")
DATA = os.path.join(ROOT, "data", "spider")
BEAM = os.path.join(REPO, "beam_outputs", "raw", "spider", "resdsql.dev.beam8.txt")
MODEL_NAME, BEAM_SIZE = "resdsql", 8

ok_all = True


def report(step, ok, detail=""):
    global ok_all
    ok_all = ok_all and ok
    print(f"[{'OK ' if ok else 'FAIL'}] {step:<34} {detail}")
    return ok


# 1 --------------------------------------------------------------------------
print("=" * 100)
print("PREFLIGHT")
print("=" * 100)

import importlib

REQUIRED = ["sqlparse", "prettytable", "nltk", "editdistance", "colorama",
            "networkx", "tqdm", "openai", "requests"]
missing = []
for mod in REQUIRED:
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f"[..] installing missing packages: {missing}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)
    missing = [m for m in missing if importlib.util.find_spec(m) is None]
versions = []
for mod in REQUIRED:
    try:
        m = importlib.import_module(mod)
        versions.append(f"{mod}={getattr(m, '__version__', '?')}")
    except ImportError:
        pass
report("1. python packages", not missing, ", ".join(versions))

# 2 --------------------------------------------------------------------------
import nltk

nltk_ok = True
for pkg, probe in (("punkt_tab", "punkt_tab"), ("punkt", "punkt")):
    try:
        nltk.data.find(f"tokenizers/{probe}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
            nltk.data.find(f"tokenizers/{probe}")
        except Exception as exc:  # noqa: BLE001
            nltk_ok = False
            print(f"      could not fetch nltk '{pkg}': {exc}")
try:
    from nltk.tokenize import word_tokenize
    word_tokenize("How many players are there?")
except Exception as exc:  # noqa: BLE001
    nltk_ok = False
    print(f"      word_tokenize still failing: {exc}")
report("2. nltk tokenizer data", nltk_ok, "punkt_tab / punkt")

# 3 --------------------------------------------------------------------------
files = {
    "dev.json": os.path.join(DATA, "dev.json"),
    "tables.json": os.path.join(DATA, "tables.json"),
}
data_ok = True
details = []
for name, path in files.items():
    if os.path.exists(path):
        details.append(f"{name}={os.path.getsize(path) // 1024}KB")
    else:
        data_ok = False
        details.append(f"{name}=MISSING")

dev = json.load(open(files["dev.json"], encoding="utf-8")) if os.path.exists(files["dev.json"]) else []
tables = json.load(open(files["tables.json"], encoding="utf-8")) if os.path.exists(files["tables.json"]) else []
db_ids = sorted({ex["db_id"] for ex in dev})
available = []
for db in db_ids:
    p = os.path.join(DATA, "database", db, f"{db}.sqlite")
    if os.path.exists(p):
        available.append((db, os.path.getsize(p)))
details.append(f"dev={len(dev)}q/{len(db_ids)}db")
details.append(f"tables={len(tables)}")
report("3. dataset files", data_ok and bool(dev) and bool(tables), ", ".join(details))
report("4. sqlite databases on disk", len(available) > 0,
       ", ".join(f"{d}({s // 1024}KB)" for d, s in available) or "NONE -- nothing to run")
if not available:
    print("\nno database available: cannot run. aborting.")
    sys.exit(1)
TARGET_DB = "wta_1" if any(d == "wta_1" for d, _ in available) else available[0][0]
report("   target db for this run", True, TARGET_DB)

# 5 --------------------------------------------------------------------------
beam_ok = os.path.exists(BEAM)
beam_detail = "MISSING"
if beam_ok:
    raw = [l.strip(";\n ") for l in open(BEAM, encoding="utf-8").readlines() if l.strip()]
    if raw and raw[0][0] != "s":
        raw = [l for i, l in enumerate(raw) if i % 9 != 0]
    raw = [l.split("|")[1].strip() if "|" in l else l for l in raw]
    beam_detail = f"{len(raw)} beams / {BEAM_SIZE} = {len(raw) / BEAM_SIZE:.0f} questions"
    beam_ok = len(raw) / BEAM_SIZE == len(dev)
report("5. beam outputs match split", beam_ok, beam_detail)

# 6 --------------------------------------------------------------------------
env_path = os.path.join(HERE, ".env")
api_key = None
if os.path.exists(env_path):
    for line in open(env_path, encoding="utf-8"):
        if line.startswith("DEEPSEEK_API_KEY="):
            api_key = line.split("=", 1)[1].strip()
api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
api_ok, api_detail = False, "no key found in work/.env"
if api_key:
    try:
        body = json.dumps({"model": "deepseek-chat",
                           "messages": [{"role": "user", "content": "ok"}],
                           "max_tokens": 3, "temperature": 0}).encode()
        req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=45) as r:
            resp = json.load(r)
        api_ok = True
        api_detail = f"reachable, served model={resp.get('model')}"
    except Exception as exc:  # noqa: BLE001
        api_detail = f"call failed: {exc}"
report("6. deepseek api", api_ok, api_detail)

# 7 --------------------------------------------------------------------------
print("-" * 100)
print("[..] 7. one-question end-to-end smoke test")
print("-" * 100)
smoke_out = os.path.join(ROOT, "outputs", "_smoke")
env = dict(os.environ, PYTHONPATH=REPO)
if api_key:
    env["DEEPSEEK_API_KEY"] = api_key
cmd = [sys.executable, os.path.join(HERE, "run_infer_llm.py"),
       "--model_name", MODEL_NAME, "--beam_size", str(BEAM_SIZE),
       "--test_file_path", files["dev.json"],
       "--beam_output_file_path", BEAM,
       "--table_file_path", files["tables.json"],
       "--db_dir", os.path.join(DATA, "database"),
       "--output_file_path", os.path.join(smoke_out, "preds.txt"),
       "--only_db", TARGET_DB, "--limit", "1",
       "--verifier_cache", os.path.join(smoke_out, "verifier_cache.jsonl")]
proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
tail = [l for l in (proc.stdout or "").splitlines() if not l.strip().endswith("it/s]")]
smoke_ok = proc.returncode == 0 and os.path.exists(os.path.join(smoke_out, "run_stats.json"))
stats = {}
if smoke_ok:
    stats = json.load(open(os.path.join(smoke_out, "run_stats.json"), encoding="utf-8"))
    # a real success means the pipeline actually built an explanation and called the verifier
    smoke_ok = stats.get("explanations_built", 0) > 0 and stats.get("llm_calls", 0) > 0
if not smoke_ok:
    print((proc.stdout or "")[-3000:])
    print((proc.stderr or "")[-3000:])
report("7. smoke test (1 question)", smoke_ok, json.dumps(stats))

print("=" * 100)
print("READY" if ok_all else "NOT READY -- fix the FAIL lines above")
print("=" * 100)
sys.exit(0 if ok_all else 1)
