#!/usr/bin/env python3
"""
fetch_eval_problems.py

Fetches 60 NEW problems from deepmind/code_contests (not in the training pool)
split into four eval categories:

  - high_memory : 20 problems tagged graphs / trees / dfs and similar / dp / data structures
  - easy        : 10 problems  800 <= cf_rating < 1200  (not already in high_memory)
  - medium      : 20 problems 1200 <= cf_rating < 1600  (not already in high_memory)
  - hard        : 10 problems cf_rating >= 1600         (not already in high_memory)

Saves to online_rl/cc_pool_cache_test.json in the same format as cc_pool_cache.json.

Usage:
    python3 online_rl/fetch_eval_problems.py [--scan 20000]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TRAIN_CACHE  = "online_rl/cc_pool_cache.json"
TEST_CACHE   = "online_rl/cc_pool_cache_test.json"

MEMORY_TAGS  = {"graphs", "trees", "dfs and similar", "dp", "data structures"}

BUCKETS = {
    "high_memory": {"cap": 20, "problems": []},
    "easy":        {"cap": 10, "problems": []},
    "medium":      {"cap": 20, "problems": []},
    "hard":        {"cap": 10, "problems": []},
}


def _all_full() -> bool:
    return all(len(b["problems"]) >= b["cap"] for b in BUCKETS.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=int, default=20000,
                        help="Max problems to scan from HuggingFace (default 20000)")
    args = parser.parse_args()

    # Load training pool to get existing IDs
    existing_ids: set[str] = set()
    if os.path.exists(TRAIN_CACHE):
        with open(TRAIN_CACHE) as f:
            train = json.load(f)
        existing_ids = {p["task_id"] for p in train.get("problems", [])}
        print(f"[fetch] training pool: {len(existing_ids)} problems to skip")

    # Also skip any problems already in the test cache
    if os.path.exists(TEST_CACHE):
        with open(TEST_CACHE) as f:
            existing_test = json.load(f)
        for p in existing_test.get("problems", []):
            existing_ids.add(p["task_id"])
        print(f"[fetch] existing test cache: will skip those too")

    from datasets import load_dataset
    ds = load_dataset("deepmind/code_contests", split="train", streaming=True)

    scanned = 0
    print(f"[fetch] scanning up to {args.scan} problems...")

    for row in ds:
        if scanned >= args.scan or _all_full():
            break
        scanned += 1

        task_id = f"cc_{row['name']}"
        if task_id in existing_ids:
            continue

        # Must have Python 3 solution
        py3 = [s for l, s in zip(row["solutions"]["language"],
                                  row["solutions"]["solution"]) if l == 3]
        if not py3:
            continue

        # Must have test inputs
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

        tl_raw     = row.get("time_limit") or {}
        time_limit = tl_raw.get("seconds", 0) if isinstance(tl_raw, dict) else float(tl_raw or 0)
        if time_limit == 0:
            continue

        cf_rating = row.get("cf_rating", 0) or 0
        cf_tags   = row.get("cf_tags", []) or []

        problem = {
            "task_id":      task_id,
            "description":  row["description"],
            "ref_solution": py3[0],
            "stdin":        stdin,
            "expected":     expected,
            "cf_rating":    cf_rating,
            "cf_tags":      cf_tags,
            "time_limit":   time_limit,
        }

        # Assign to first bucket that needs it
        tags_set = set(cf_tags)
        if len(BUCKETS["high_memory"]["problems"]) < BUCKETS["high_memory"]["cap"]:
            if tags_set & MEMORY_TAGS:
                problem["eval_category"] = "high_memory"
                BUCKETS["high_memory"]["problems"].append(problem)
                existing_ids.add(task_id)
                print(f"[fetch] high_memory ({len(BUCKETS['high_memory']['problems'])}/20): {task_id} | rating={cf_rating} | tags={cf_tags}")
                continue

        if len(BUCKETS["easy"]["problems"]) < BUCKETS["easy"]["cap"]:
            if 800 <= cf_rating < 1200:
                problem["eval_category"] = "easy"
                BUCKETS["easy"]["problems"].append(problem)
                existing_ids.add(task_id)
                print(f"[fetch] easy ({len(BUCKETS['easy']['problems'])}/10): {task_id} | rating={cf_rating}")
                continue

        if len(BUCKETS["medium"]["problems"]) < BUCKETS["medium"]["cap"]:
            if 1200 <= cf_rating < 1600:
                problem["eval_category"] = "medium"
                BUCKETS["medium"]["problems"].append(problem)
                existing_ids.add(task_id)
                print(f"[fetch] medium ({len(BUCKETS['medium']['problems'])}/20): {task_id} | rating={cf_rating}")
                continue

        if len(BUCKETS["hard"]["problems"]) < BUCKETS["hard"]["cap"]:
            if cf_rating >= 1600:
                problem["eval_category"] = "hard"
                BUCKETS["hard"]["problems"].append(problem)
                existing_ids.add(task_id)
                print(f"[fetch] hard ({len(BUCKETS['hard']['problems'])}/10): {task_id} | rating={cf_rating}")
                continue

        if scanned % 1000 == 0:
            counts = {k: len(v["problems"]) for k, v in BUCKETS.items()}
            print(f"[fetch] scanned {scanned} | buckets: {counts}")

    # Summary
    print(f"\n[fetch] scanned {scanned} total problems")
    all_problems = []
    for cat, bucket in BUCKETS.items():
        n = len(bucket["problems"])
        print(f"  {cat}: {n}/{bucket['cap']}")
        all_problems.extend(bucket["problems"])

    if len(all_problems) < 60:
        print(f"\n[fetch] WARNING: only collected {len(all_problems)}/60 problems — "
              f"try --scan with a larger value")

    # Save
    out = {
        "meta": {
            "source": "fetch_eval_problems.py",
            "total": len(all_problems),
            "categories": {k: len(v["problems"]) for k, v in BUCKETS.items()},
        },
        "problems": all_problems,
    }
    with open(TEST_CACHE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[fetch] saved {len(all_problems)} problems to {TEST_CACHE}")


if __name__ == "__main__":
    main()
