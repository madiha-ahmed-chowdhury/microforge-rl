#!/usr/bin/env python3
"""
fetch_training_problems.py

Fetches 600 NEW problems from deepmind/code_contests and appends them
to the training pool (cc_pool_cache.json). Skips problems already in
the training pool or eval pool.

Buckets:
  dp_graph   : 200  — tags intersect {dp, dynamic programming, matrices,
                       graphs, trees, dfs and similar, data structures}
  hard       : 200  — CF rating 1600–2200  (not already in dp_graph)
  medium_easy: 200  — CF rating  800–1600  (not already in dp_graph / hard)

Usage:
    python3 online_rl/fetch_training_problems.py [--scan 50000]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TRAIN_CACHE = "online_rl/cc_pool_cache.json"
EVAL_CACHE  = "online_rl/cc_pool_cache_test.json"

TARGET_TAGS = {
    "dp", "dynamic programming", "matrices",
    "graphs", "trees", "dfs and similar", "data structures",
}

BUCKETS = {
    "dp_graph":    {"cap": 200, "problems": []},
    "hard":        {"cap": 200, "problems": []},
    "medium_easy": {"cap": 200, "problems": []},
}


def _all_full() -> bool:
    return all(len(b["problems"]) >= b["cap"] for b in BUCKETS.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=int, default=50000,
                        help="Max problems to scan from HuggingFace (default 50000)")
    args = parser.parse_args()

    # Build set of already-known task_ids (training + eval)
    existing_ids: set[str] = set()

    with open(TRAIN_CACHE) as f:
        train = json.load(f)
    existing_problems = train.get("problems", [])
    existing_ids = {p["task_id"] for p in existing_problems}
    print(f"[fetch] training pool: {len(existing_ids)} problems to skip")

    if os.path.exists(EVAL_CACHE):
        with open(EVAL_CACHE) as f:
            eval_data = json.load(f)
        for p in eval_data.get("problems", []):
            existing_ids.add(p["task_id"])
        print(f"[fetch] eval pool: will skip those too ({len(existing_ids)} total skipped)")

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

        # Must have a time limit
        tl_raw     = row.get("time_limit") or {}
        time_limit = tl_raw.get("seconds", 0) if isinstance(tl_raw, dict) else float(tl_raw or 0)
        if time_limit == 0:
            continue

        cf_rating = row.get("cf_rating", 0) or 0
        cf_tags   = row.get("cf_tags", []) or []
        tags_set  = set(cf_tags)

        problem = {
            "task_id":            task_id,
            "description":        row["description"],
            "ref_solution":       py3[0],
            "stdin":              stdin,
            "expected":           expected,
            "cf_rating":          cf_rating,
            "cf_tags":            cf_tags,
            "time_limit":         time_limit,
            "test_case_generator": "",
        }

        # Bucket 1: dp/graph/matrix tags (priority)
        if len(BUCKETS["dp_graph"]["problems"]) < BUCKETS["dp_graph"]["cap"]:
            if tags_set & TARGET_TAGS:
                BUCKETS["dp_graph"]["problems"].append(problem)
                existing_ids.add(task_id)
                n = len(BUCKETS["dp_graph"]["problems"])
                print(f"[fetch] dp_graph ({n}/200): {task_id} | rating={cf_rating} | tags={cf_tags}")
                continue

        # Bucket 2: hard (1600–2200), not already taken
        if len(BUCKETS["hard"]["problems"]) < BUCKETS["hard"]["cap"]:
            if 1600 <= cf_rating <= 2200:
                BUCKETS["hard"]["problems"].append(problem)
                existing_ids.add(task_id)
                n = len(BUCKETS["hard"]["problems"])
                print(f"[fetch] hard ({n}/200): {task_id} | rating={cf_rating}")
                continue

        # Bucket 3: medium-easy (800–1600), not already taken
        if len(BUCKETS["medium_easy"]["problems"]) < BUCKETS["medium_easy"]["cap"]:
            if 800 <= cf_rating <= 1600:
                BUCKETS["medium_easy"]["problems"].append(problem)
                existing_ids.add(task_id)
                n = len(BUCKETS["medium_easy"]["problems"])
                print(f"[fetch] medium_easy ({n}/200): {task_id} | rating={cf_rating}")
                continue

        if scanned % 2000 == 0:
            counts = {k: len(v["problems"]) for k, v in BUCKETS.items()}
            print(f"[fetch] scanned {scanned} | buckets: {counts}")

    # Summary
    print(f"\n[fetch] scanned {scanned} total problems")
    new_problems = []
    for cat, bucket in BUCKETS.items():
        n = len(bucket["problems"])
        print(f"  {cat}: {n}/{bucket['cap']}")
        new_problems.extend(bucket["problems"])

    if len(new_problems) < 600:
        print(f"\n[fetch] WARNING: only got {len(new_problems)}/600 — try --scan with a larger value")

    # Append to training pool
    all_problems = existing_problems + new_problems
    train["problems"] = all_problems
    train["meta"]["total"] = len(all_problems)
    train["meta"]["last_expanded"] = f"added {len(new_problems)} problems (dp_graph/hard/medium_easy)"

    with open(TRAIN_CACHE, "w") as f:
        json.dump(train, f, indent=2)

    print(f"\n[fetch] training pool: {len(existing_problems)} → {len(all_problems)} problems")
    print(f"[fetch] saved to {TRAIN_CACHE}")


if __name__ == "__main__":
    main()
