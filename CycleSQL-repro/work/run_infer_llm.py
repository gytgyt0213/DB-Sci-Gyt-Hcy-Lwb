"""CycleSQL inference with the NLI verifier replaced by an LLM.

WHY THIS FILE EXISTS
--------------------
CycleSQL's original `scripts/run_infer.py` loads a *fine-tuned T5-large* sequence
classifier from `saved_models/checkpoint-500` and uses it as the entailment
verifier of the feedback loop. That checkpoint is distributed **only** through a
Google Drive link (https://drive.google.com/file/d/1Efe3zsh9HaxX3FVhu7SvwynLFs6KJaLm/view),
which is unreachable from this machine, and it is ~2.95 GB (over our 1 GB budget).
The repository also does not ship the NLI training data (`data/nli/train.json`)
nor the training beam outputs, so the checkpoint cannot be retrained either.

Therefore we keep the *entire rest of the pipeline byte-for-byte identical*
(SQL rewriting, provenance extraction, semantics annotation, template-based
explanation generation, beam iteration, fallback) and substitute only the
entailment decision with a DeepSeek chat completion. Every place where we
deviate from the original is marked with `# [PATCH]`.

ORIGINAL VERIFIER (scripts/run_infer.py, lines 154-162):
    text = f"premise: (sql) {query} | (query result) {result} | (explanation) {exp} hypothesis: {nl}"
    input = tokenizer(text, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = nli_model(**input).logits
        predicted_class_id = logits.argmax().item()
        if predicted_class_id == 0:   # entailment
            pred = sql

We reuse the exact same premise/hypothesis string as the LLM prompt.

ADDITIONAL DEVIATION: `--per-db` / `--limit` subsampling. The original iterates
`data` positionally and slices the beam file with `idx * beam_size`. To allow a
stratified subset we keep the original index of every example and slice the
beam file by that original index, which reproduces the original behaviour
exactly when no subsampling is requested.
"""
from copy import deepcopy
import argparse
import hashlib
import json
import os
import random
import sqlite3
import sys
import threading
import time
import traceback

# [PATCH] `src/utils.py:436` uses `Thread.isAlive()`, which was deprecated in
# Python 3.8 and REMOVED in 3.9. The artifact therefore only runs on <=3.8.
# We keep the original sources untouched and alias the modern name instead.
if not hasattr(threading.Thread, "isAlive"):
    threading.Thread.isAlive = threading.Thread.is_alive
from typing import Union

from openai import OpenAI
from prettytable import PrettyTable
from tqdm import tqdm

from spider_utils.evaluation import (
    build_foreign_key_map_from_json,
    build_valid_col_units,
    eval_exec_match1,
    rebuild_sql_col,
    rebuild_sql_val,
)
from spider_utils.process_sql import Schema, get_schema, get_sql
from spider_utils.utils import read_single_dataset_schema_from_database

from src.explainer import Explainer
from src.utils import (
    EXCLUDE_KEYWORDS,
    clause2dialect,
    execute_sqlite_query,
    find_relationship_tables,
    get_db_statistics,
    get_one_query_result,
    parse_query_semantics,
    remove_join_conditions,
    rewrite_asterisk_query,
    sql_format,
)

# --------------------------------------------------------------------------
# [PATCH] LLM verifier — replaces the fine-tuned T5 NLI classifier
# --------------------------------------------------------------------------

NLI_SYSTEM_PROMPT = (
    "You are a strict natural language inference (NLI) classifier used inside a "
    "text-to-SQL pipeline. You are given a PREMISE made of three parts: a candidate "
    "SQL query, a sample of the result that query returns, and a natural-language "
    "explanation of that result. You are given a HYPOTHESIS: the user's original "
    "natural-language question.\n"
    "Decide whether the PREMISE entails the HYPOTHESIS, i.e. whether the SQL query "
    "faithfully answers the hypothesis, as witnessed by the result and the "
    "explanation. Be strict: a query that answers a different question, filters on "
    "the wrong values, or covers the wrong set of rows does NOT entail the hypothesis.\n"
    "Answer with exactly one token: ENTAILMENT or NOT_ENTAILMENT."
)


class LLMVerifier:
    """Cache-backed entailment oracle standing in for the T5 classifier."""

    def __init__(self, api_key: str, model: str, base_url: str, cache_path: str,
                 max_retries: int = 4):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._cache_path = cache_path
        self._max_retries = max_retries
        self._cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec["entailment"]
        self._cache_fh = open(cache_path, "a", encoding="utf-8")
        self.calls = 0
        self.cache_hits = 0
        self.errors = 0

    @staticmethod
    def _key(premise: str, hypothesis: str) -> str:
        return hashlib.sha1((premise + "\x00" + hypothesis).encode("utf-8")).hexdigest()

    def entails(self, sql: str, result: str, explanation: str, question: str) -> bool:
        # Identical composition to the original verifier input.
        premise = f"(sql) {sql} | (query result) {result} | (explanation) {explanation}"
        hypothesis = question
        key = self._key(premise, hypothesis)

        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        user_prompt = f"premise: {premise} hypothesis: {hypothesis}"
        last_err = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": NLI_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=8,
                )
                self.calls += 1
                text = (resp.choices[0].message.content or "").strip().upper()
                # NOTE: "ENTAILMENT" is a substring of "NOT_ENTAILMENT" -> test NOT first.
                entailment = ("NOT_ENTAILMENT" not in text) and ("ENTAILMENT" in text)
                self._cache[key] = entailment
                self._cache_fh.write(json.dumps({"key": key, "entailment": entailment,
                                                 "response": text}) + "\n")
                self._cache_fh.flush()
                return entailment
            except Exception as exc:  # noqa: BLE001 - report and retry
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        self.errors += 1
        raise RuntimeError(f"LLM verifier failed after {self._max_retries} attempts: {last_err}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--beam_size", type=int, required=True)
    p.add_argument("--test_file_path", type=str, required=True)
    p.add_argument("--beam_output_file_path", type=str, required=True)
    p.add_argument("--table_file_path", type=str, required=True)
    p.add_argument("--db_dir", type=str, required=True)
    p.add_argument("--output_file_path", type=str, required=True)
    # [PATCH] replaces --nli_model_dir
    p.add_argument("--llm_model", type=str, default="deepseek-chat")
    p.add_argument("--llm_base_url", type=str, default="https://api.deepseek.com")
    p.add_argument("--llm_api_key", type=str, default=None)
    p.add_argument("--verifier_cache", type=str, default="outputs/verifier_cache.jsonl")
    # [PATCH] subsampling (0 = use the whole split)
    p.add_argument("--limit", type=int, default=0, help="first N examples")
    p.add_argument("--per_db", type=int, default=0, help="up to N examples per database")
    p.add_argument("--only_db", type=str, default=None,
                   help="restrict to one db_id (useful when only some sqlite files are on disk)")
    # [PATCH] the original calls unseeded `random.choice` when picking which
    # provenance row to explain (src/utils.py:get_one_query_result), so two runs
    # of the same command produce different explanations and different accuracy.
    # Seeding makes the reproduction reproducible at all.
    p.add_argument("--seed", type=int, default=0, help="-1 disables seeding (original behaviour)")
    return p.parse_args()


def select_indices(data, limit, per_db, only_db=None):
    if only_db:
        picked = [i for i, ex in enumerate(data) if ex["db_id"] == only_db]
    elif per_db > 0:
        seen = {}
        picked = []
        for i, ex in enumerate(data):
            n = seen.get(ex["db_id"], 0)
            if n < per_db:
                seen[ex["db_id"]] = n + 1
                picked.append(i)
    else:
        picked = list(range(len(data)))
    if limit > 0:            # `limit` now caps whatever the other selectors picked,
        picked = picked[:limit]   # so `--only_db X --limit 1` is a 1-question smoke test
    return picked


def main():
    args = parse_args()

    api_key = args.llm_api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("no API key: pass --llm_api_key or set DEEPSEEK_API_KEY")

    kmaps = build_foreign_key_map_from_json(args.table_file_path)
    id2schema = {ex["db_id"]: ex for ex in json.load(open(args.table_file_path, encoding="utf-8"))}
    data = json.load(open(args.test_file_path, encoding="utf-8"))

    out_dir = os.path.dirname(args.output_file_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(args.beam_output_file_path))[0]
    beam_serialization_file = os.path.join(out_dir, f"{file_name}.json")
    nl2tuple = {}
    if os.path.exists(beam_serialization_file):
        for ex in json.load(open(beam_serialization_file, encoding="utf-8")):
            nl2tuple[(ex["question"], ex["query"])] = (ex["explanation"], ex["result"])

    # ---- beam de-serialisation: identical to the original --------------------
    beam_preds = [l.strip(";\n ") for l in open(args.beam_output_file_path, encoding="utf-8").readlines() if l.strip()]
    if "resdsql" in args.model_name:
        if beam_preds[0][0] != "s":  # exclude the natural-language question line
            beam_preds = [l for idx, l in enumerate(beam_preds) if idx % 9 != 0]
        beam_preds = [l.split("|")[1].strip() if "|" in l else l for l in beam_preds]
    elif args.model_name == "smbop":
        beam_preds = [l.split("\t")[0].strip() for l in beam_preds]
    assert len(beam_preds) / args.beam_size == len(data), \
        f"beam file/model mismatch: {len(beam_preds)} lines / {args.beam_size} != {len(data)} questions"

    def beams_for(orig_idx):
        return beam_preds[orig_idx * args.beam_size:(orig_idx + 1) * args.beam_size]

    if args.seed >= 0:                      # [PATCH] see --seed
        random.seed(args.seed)

    indices = select_indices(data, args.limit, args.per_db, args.only_db)
    print(f"[info] {len(indices)} of {len(data)} dev questions selected "
          f"({len({data[i]['db_id'] for i in indices})} databases)")

    verifier = LLMVerifier(api_key, args.llm_model, args.llm_base_url, args.verifier_cache)
    explainer = Explainer(args.db_dir)

    prev_db_id = None
    outputs, outputs2 = [], []          # outputs2: the (sql, explanation, result) triples actually built
    n_switched = 0                      # questions where the verifier moved us off beam #1
    _errs = {"n": 0, "last": "", "kinds": {}}   # [PATCH] diagnostics for swallowed exceptions
    baseline_top1 = []                  # beam #1 for every selected question (the "no feedback loop" baseline)

    for orig_idx in tqdm(indices, total=len(indices)):
        ex = data[orig_idx]
        gold_idx = -1

        db_id = ex["db_id"]
        if db_id != prev_db_id:
            db_file = os.path.join(args.db_dir, db_id, f"{db_id}.sqlite")
            conn = sqlite3.connect(db_file)
            conn.text_factory = lambda b: b.decode(errors="ignore")
            cursor = conn.cursor()
            schema = read_single_dataset_schema_from_database(id2schema[db_id], cursor)
            schema2ids = Schema(get_schema(db_file))
            relation2tables = find_relationship_tables(schema)
            prev_db_id = db_id
        nl = ex["question"]
        gold = ex["query"]
        try:
            g_sql, _ = get_sql(schema2ids, gold)
        except Exception:
            pass

        sqls = []
        pred = None
        beams = beams_for(orig_idx)
        baseline_top1.append(beams[0])

        # identical to the original: locate the gold-matching beam (used only for
        # the debug printout further down, never to set `pred`)
        for idx, sql in enumerate(beams):
            try:
                p_sql, _ = get_sql(schema2ids, sql)
                if eval_exec_match1(db_file, sql, gold, deepcopy(p_sql), deepcopy(g_sql)) == 1:
                    gold_idx = idx
                    break
            except Exception:
                continue

        for idy, sql in enumerate(beams):
            try:
                if nl2tuple:
                    if not (nl, sql) in nl2tuple.keys():
                        continue
                    exp, result = nl2tuple[(nl, sql)]
                    query = remove_join_conditions(sql_format(sql))
                else:
                    p_sql, _ = get_sql(schema2ids, sql)
                    rewrite_sql = rewrite_asterisk_query(sql, schema, db_file)
                    p_rewrite_sql, table2alias = get_sql(schema2ids, rewrite_sql)
                    result_set: Union[list, list, list] = execute_sqlite_query(rewrite_sql, p_rewrite_sql, db_file)
                    if not result_set[1]:
                        continue
                    table2statistics = get_db_statistics(sql, p_sql, table2alias, schema, cursor)
                    join_semantics, group_semantics = parse_query_semantics(p_sql, schema, relation2tables, table2statistics)

                    filtering_modifier: str = clause2dialect(p_sql, schema, group_semantics, cls_type="where", prefix="filtered by ", bracket=True)
                    filtering_modifier += clause2dialect(p_sql, schema, group_semantics, cls_type="having", prefix=" and ", bracket=True) if filtering_modifier else \
                        clause2dialect(p_sql, schema, group_semantics, cls_type="having", prefix="filtered by ", bracket=True)
                    sorting_modifier: str = clause2dialect(p_sql, schema, group_semantics, cls_type="orderBy", prefix="sorted ", bracket=True)
                    exp = f'The query returns a result set with {len(result_set[1])} column{"s" if len(result_set[1]) > 1 else ""} ' + \
                        f'[{", ".join([f"{c}" if any(kw in "".join(c.split()).lower() for kw in EXCLUDE_KEYWORDS) or t == "NoneType" else f"{c} ({t})" for c, t in zip(result_set[1], result_set[2])])}] and ' + \
                        f'{len([r for r in result_set[0] if any(c for c in r)])} row{"s" if len(result_set[0]) > 1 else ""}{", " if filtering_modifier else ""}' + \
                        f'{filtering_modifier if filtering_modifier else ""}{", " if sorting_modifier else ""}{sorting_modifier if sorting_modifier else ""}.'

                    if not result_set[0] or all(r is None for r in result_set[0][0]):
                        result = ""
                    else:
                        exp += f'{" Among them, t" if len(result_set[0]) > 1 else " T"}he result, {"for example, " if len(result_set[0]) > 1 else ""}'
                        to_explain_result: PrettyTable = get_one_query_result(result_set, p_rewrite_sql, None)
                        explainer.set_database_context(db_id, schema, schema2ids, relation2tables, table2statistics, table2alias, join_semantics, group_semantics)
                        explainer.set_context(question=nl, sql=rewrite_sql, parsed_sql=p_rewrite_sql, result=to_explain_result)
                        exp += explainer.explain()
                        result = ", ".join([f"{field}: {value}" for field, value in
                                            zip(to_explain_result.field_names, to_explain_result.rows[0])]).strip()

                    query = remove_join_conditions(sql_format(sql))
                    if query in sqls:
                        continue
                    sqls.append(query)

                    outputs2.append({"db_id": db_id, "question": nl, "query": sql,
                                     "explanation": exp, "result": result})

                # ============ [PATCH] verification: LLM instead of T5 ============
                if verifier.entails(query, result, exp, nl):
                    pred = sql
                    if idy != 0:
                        n_switched += 1
                    if eval_exec_match1(db_file, sql, gold, deepcopy(p_sql), deepcopy(g_sql)) != 1 and gold_idx > -1:
                        print(f"\nNL query: {nl}\nPredicted SQL index: {idy}\nPredicted SQL: {sql}\n"
                              f"Predicted SQL Explanation:{exp}\nResult: {result}")
                        print(f"Gold SQL Index: {gold_idx}\nGold SQL: {gold}\nGold SQL Explanation: {exp}\nResult: {result}")
                    break
                # ============ end [PATCH] ========================================
            except Exception as exc:  # noqa: BLE001
                # [PATCH] the original swallows every exception with a bare
                # `except: continue`, which makes silent total failure invisible.
                # Keep the control flow, but surface the first few tracebacks and
                # count every distinct failure kind.
                kind = f"{type(exc).__name__}: {exc}"
                _errs["kinds"][kind] = _errs["kinds"].get(kind, 0) + 1
                if _errs["n"] < 3:
                    traceback.print_exc()
                _errs["n"] += 1
                _errs["last"] = kind
                continue

        if not pred:
            add = False
            for sql in beams:
                try:
                    p_sql, _ = get_sql(schema2ids, sql)
                    p_valid_col_units = build_valid_col_units(p_sql["from"]["table_units"], schema2ids)
                    p_sql = rebuild_sql_val(p_sql)
                    p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmaps[db_id])
                    cursor.execute(sql)
                    outputs.append(sql)
                    add = True
                    break
                except Exception:
                    continue
            if not add:
                outputs.append(beams[0])
        else:
            outputs.append(pred)

    if not os.path.exists(beam_serialization_file):
        with open(beam_serialization_file, "w", encoding="utf-8") as f:
            json.dump(outputs2, f, indent=4)

    with open(args.output_file_path, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(f"{o}\n")

    # baseline = beam #1 for the very same questions (what the feedback loop has to beat)
    baseline_path = os.path.join(out_dir, "preds_baseline_top1.txt")
    with open(baseline_path, "w", encoding="utf-8") as f:
        for o in baseline_top1:
            f.write(f"{o}\n")

    verifier._cache_fh.close()
    stats = {
        "questions": len(indices),
        "llm_calls": verifier.calls,
        "llm_cache_hits": verifier.cache_hits,
        "llm_errors": verifier.errors,
        "verifier_switched_off_beam1": n_switched,
        "explanations_built": len(outputs2),
        "swallowed_exceptions": _errs["n"],
        "last_exception": _errs["last"],
        "exception_kinds": _errs["kinds"],
        "preds": args.output_file_path,
        "baseline_top1": baseline_path,
    }
    with open(os.path.join(out_dir, "run_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
