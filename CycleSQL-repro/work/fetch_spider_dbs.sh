#!/bin/bash
# Fetch the Spider sqlite databases that the dev split needs.
# Source mirror: hf-mirror.com (huggingface.co is unreachable from this host).
# Layout produced: data/spider/database/<db_id>/<db_id>.sqlite  (what CycleSQL expects)
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd -W)"   # Windows-style path: Git Bash's /e/... is not understood by python.exe
DEV_JSON="$ROOT/data/spider/dev.json"
DB_ROOT="$ROOT/data/spider/database"
BASE="https://hf-mirror.com/datasets/ashishkattamuri/multispider-processed/resolve/main/databases"

mkdir -p "$DB_ROOT"

python_exe="${PY:-F:/pythonnewversion/cyclesql/python.exe}"

DB_IDS=$("$python_exe" -c "
import json,sys
d=json.load(open(r'$DEV_JSON',encoding='utf-8'))
print('\n'.join(sorted({x['db_id'] for x in d})))
")

for db in $DB_IDS; do
    out="$DB_ROOT/$db/$db.sqlite"
    if [ -s "$out" ]; then
        echo "SKIP  $db"
        continue
    fi
    mkdir -p "$DB_ROOT/$db"
    ok=0
    # The mirror sometimes accepts the TCP connection and then stalls forever, so a
    # long -m would burn minutes per attempt. Keep the timeout short and retry a lot.
    for try in $(seq 1 12); do
        code=$(curl -sL --connect-timeout 10 -m 30 -o "$out.tmp" -w "%{http_code}" "$BASE/$db.sqlite")
        if [ "$code" = "200" ] && [ -s "$out.tmp" ]; then mv "$out.tmp" "$out"; ok=1; break; fi
        rm -f "$out.tmp"
        sleep 2
    done
    if [ "$ok" = "1" ]; then
        echo "OK    $db  $(stat -c%s "$out") bytes"
    else
        echo "FAIL  $db  (last http $code)"
        rm -f "$out"
    fi
    sleep 1
done
