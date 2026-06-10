import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from online_rl.config import PATHS, FEATURE_COLS

_CC_POOL_CACHE = "online_rl/cc_pool_cache.json"


def _load_state_map() -> dict:
    state_map = {}
    try:
        with open(PATHS["raw_transitions"]) as f:
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line)
                tid = t.get("task_id")
                if tid and tid not in state_map:
                    state_map[tid] = t.get("state", {})
    except FileNotFoundError:
        pass
    return state_map


def _build_default_state(row: dict) -> dict:
    from actions import static_analyse, build_state
    code = row.get("code", "")
    cf   = static_analyse(code)
    return build_state(row.get("prompt", ""), cf, row.get("task_type", 3))


def build_from_hf(heavy_only: bool = False) -> list:
    from actions import _load_effibench_hf
    state_map = _load_state_map()

    dataset = "effibench_large" if heavy_only else "effibench"
    rows    = _load_effibench_hf(large=heavy_only)
    print(f"HuggingFace rows loaded: {len(rows)}")

    inputs  = []
    skipped = 0

    for row in rows:
        code = row.get("code", "")
        if not code:
            skipped += 1
            continue

        test_cases = row.get("test_cases", [])
        stdin    = test_cases[0].get("input", "")  if test_cases else ""
        expected = test_cases[0].get("output", "") if test_cases else ""
        generator = row.get("test_case_generator", "")

        task_id = row.get("task_id", "")
        state   = state_map.get(task_id) or _build_default_state(row)

        inputs.append({
            "task_id":              task_id,
            "code":                 code,
            "stdin":                stdin,
            "expected":             expected,
            "test_case_generator":  generator,
            "prompt":               row.get("prompt", ""),
            "stress_level":         row.get("stress_level", "low"),
            "state":                {k: state.get(k, 0) for k in FEATURE_COLS},
        })

    return inputs, skipped


def build_from_pairs() -> list:
    state_map = _load_state_map()

    with open(PATHS["code_pairs"]) as f:
        first = json.loads(f.readline())
    print("code_pairs keys:", list(first.keys()))

    with open(PATHS["code_pairs"]) as f:
        records = [json.loads(line) for line in f if line.strip()]

    inputs  = []
    skipped = 0

    for rec in records:
        code = rec.get("refined_code", "") or rec.get("generated_code", "")
        if not code:
            skipped += 1
            continue

        task_id = rec.get("task_id", "")
        state   = state_map.get(task_id, {})
        if not state:
            skipped += 1
            continue

        test_cases = rec.get("test_cases", [])
        stdin    = rec.get("stress_input", "") or (test_cases[0].get("input", "")  if test_cases else "")
        expected = rec.get("ref_output", "")   or (test_cases[0].get("output", "") if test_cases else "")

        inputs.append({
            "task_id":             task_id,
            "code":                code,
            "description":         rec.get("prompt", ""),
            "stdin":               stdin,
            "expected":            expected,
            "test_case_generator": "",
            "stress_level":        "unknown",
            "source":              "effibench",
            "state":               {k: state.get(k, 0) for k in FEATURE_COLS},
        })

    return inputs, skipped


def add_new_to_pool(n: int, min_rating: int, max_rating: int) -> None:
    existing = []
    if os.path.exists(_CC_POOL_CACHE):
        with open(_CC_POOL_CACHE) as f:
            data = json.load(f)
        existing = data.get("problems", [])
    existing_ids = {p["task_id"] for p in existing}
    print(f"[add] existing problems in pool: {len(existing)}")

    from datasets import load_dataset
    ds = load_dataset("deepmind/code_contests", split="train", streaming=True)
    candidates = []
    scanned = 0

    for row in ds:
        scanned += 1
        cf = row["cf_rating"]
        if not (min_rating <= cf <= max_rating):
            continue

        task_id = f"cc_{row['name']}"
        if task_id in existing_ids:
            continue

        py3 = [s for l, s in zip(row["solutions"]["language"], row["solutions"]["solution"]) if l == 3]
        if not py3:
            continue

        gen_in  = row["generated_tests"]["input"]
        gen_out = row["generated_tests"]["output"]
        pub_in  = row["public_tests"]["input"]
        pub_out = row["public_tests"]["output"]

        if gen_in:
            idx      = max(range(len(gen_in)), key=lambda i: len(gen_in[i]))
            stdin    = gen_in[idx]
            expected = gen_out[idx] if idx < len(gen_out) else ""
        elif pub_in:
            stdin    = pub_in[0]
            expected = pub_out[0] if pub_out else ""
        else:
            continue

        candidates.append({
            "task_id":             task_id,
            "description":         row["description"],
            "ref_solution":        py3[0],
            "stdin":               stdin,
            "expected":            expected,
            "cf_rating":           cf,
            "cf_tags":             row["cf_tags"],
            "test_case_generator": "",
        })

    print(f"[add] scanned {scanned} rows, found {len(candidates)} new candidates "
          f"(rating {min_rating}-{max_rating})")

    if not candidates:
        print("[add] no new problems found — nothing to add")
        return

    random.seed(int(time.time()))
    sampled = random.sample(candidates, min(n, len(candidates)))
    if len(candidates) < n:
        print(f"[add] only {len(candidates)} candidates available — adding all of them")

    updated = existing + sampled
    os.makedirs(os.path.dirname(_CC_POOL_CACHE), exist_ok=True)
    with open(_CC_POOL_CACHE, "w") as f:
        json.dump({"meta": {"min_rating": min_rating, "max_rating": max_rating},
                   "problems": updated}, f, indent=2)

    print(f"added {len(sampled)} new problems")
    print(f"total problems now {len(updated)}")
    print(f"new problems needing LLM call: {len(sampled)}")
    print(f"run runner.py with --resume to continue training")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",      choices=["hf", "pairs"], default="pairs",
                        help="hf=HuggingFace EffiBench, pairs=collected code_pairs file")
    parser.add_argument("--heavy-only",  action="store_true",
                        help="with --source hf: skip low-stress problems")
    parser.add_argument("--add-new",     type=int, default=None, dest="add_new",
                        help="Add N new CodeContests problems to cc_pool_cache.json")
    parser.add_argument("--min-rating",  type=int, default=1400,
                        help="Min CF rating filter for --add-new (default 1400)")
    parser.add_argument("--max-rating",  type=int, default=1800,
                        help="Max CF rating filter for --add-new (default 1800)")
    args = parser.parse_args()

    if args.add_new is not None:
        add_new_to_pool(args.add_new, args.min_rating, args.max_rating)
        return

    if args.source == "hf":
        inputs, skipped = build_from_hf(heavy_only=args.heavy_only)
    else:
        inputs, skipped = build_from_pairs()

    out_path = PATHS["inputs"]
    with open(out_path, "w") as f:
        json.dump(inputs, f, indent=2)

    has_stdin = sum(1 for i in inputs if i.get("stdin", "").strip())
    has_gen   = sum(1 for i in inputs if i.get("test_case_generator", "").strip())
    total     = len(inputs) + skipped

    print(f"total={total}  skipped={skipped}  saved={len(inputs)}")
    print(f"  with stdin:     {has_stdin}/{len(inputs)}")
    print(f"  with generator: {has_gen}/{len(inputs)}")
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
