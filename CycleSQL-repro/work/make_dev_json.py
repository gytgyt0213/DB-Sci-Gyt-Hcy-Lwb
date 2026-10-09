"""Convert the HuggingFace `xlangai/spider` validation parquet into the standard
Spider `dev.json` format that CycleSQL / test-suite-sql-eval expect.

Input : <parquet>  (validation split, 1034 rows)
Output: <out_json> list of {"db_id", "question", "query", "question_toks", "query_toks"}
"""
import json
import sys

import pyarrow.parquet as pq


def main() -> None:
    parquet_path, out_path = sys.argv[1], sys.argv[2]
    table = pq.read_table(parquet_path)
    cols = table.column_names
    print("columns:", cols, file=sys.stderr)

    records = table.to_pylist()
    out = []
    for r in records:
        item = {
            "db_id": r["db_id"],
            "question": r["question"],
            "query": r["query"],
        }
        # keep token fields if the split carries them
        for key in ("question_toks", "query_toks"):
            if key in r and r[key] is not None:
                item[key] = list(r[key])
        out.append(item)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)

    db_ids = sorted({r["db_id"] for r in out})
    print(f"wrote {len(out)} examples covering {len(db_ids)} databases -> {out_path}",
          file=sys.stderr)
    print("db_ids:", ",".join(db_ids), file=sys.stderr)


if __name__ == "__main__":
    main()
