#!/usr/bin/env python3
"""
fetch_reserve_and_gen.py

Fetches the 97 reserve problems from HuggingFace (all splits)
and stores them in cc_pool_cache.json.

Usage:
    python3 online_rl/fetch_reserve_and_gen.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE_PATH   = "online_rl/cc_pool_cache.json"
RESERVE_PATH = "online_rl/stress_candidates_reserve.txt"


def read_reserve() -> list:
    ids = []
    with open(RESERVE_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return ids


def fetch_from_hf(target_ids: list) -> dict:
    from datasets import load_dataset

    target_set = set(target_ids)
    found = {}

    for split in ["train", "test", "valid"]:
        if not target_set:
            break
        print(f"\n[fetch] scanning split={split} ({len(target_set)} remaining)...")
        try:
            ds = load_dataset("deepmind/code_contests", split=split, streaming=True)
        except Exception as e:
            print(f"[fetch] split {split} unavailable: {e}")
            continue

        for row in ds:
            task_id = f"cc_{row['name']}"
            if task_id not in target_set:
                continue

            py3 = [s for l, s in zip(
                row["solutions"]["language"],
                row["solutions"]["solution"],
            ) if l == 3]

            if not py3:
                print(f"  [skip] {task_id} — no Python 3 solution")
                target_set.discard(task_id)
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
                print(f"  [skip] {task_id} — no test inputs")
                target_set.discard(task_id)
                continue

            found[task_id] = {
                "task_id":      task_id,
                "description":  row["description"],
                "ref_solution": py3[0],
                "stdin":        stdin,
                "expected":     expected,
                "cf_rating":    row.get("cf_rating", 0),
                "cf_tags":      row.get("cf_tags", []),
            }
            target_set.discard(task_id)
            print(f"  [found] {task_id} (cf={row.get('cf_rating',0)}, stdin={len(stdin)}c)")

            if not target_set:
                break

    if target_set:
        print(f"\n[fetch] {len(target_set)} IDs not found in any split:")
        for t in sorted(target_set):
            print(f"  {t}")

    return found


def main():
    reserve = read_reserve()
    print(f"Reserve: {len(reserve)} problem IDs")

    cache    = json.load(open(CACHE_PATH))
    existing = {p["task_id"] for p in cache["problems"]}

    to_fetch = [t for t in reserve if t not in existing]
    print(f"Already in cache: {len(reserve) - len(to_fetch)}")
    print(f"Need to fetch:    {len(to_fetch)}")

    if not to_fetch:
        print("Nothing to do.")
        return

    found = fetch_from_hf(to_fetch)

    for task_id, prob in found.items():
        cache["problems"].append(prob)

    json.dump(cache, open(CACHE_PATH, "w"), indent=2)
    print(f"\n[done] added {len(found)} problems. Cache now has {len(cache['problems'])} total.")

    not_found = set(to_fetch) - set(found.keys())
    if not_found:
        print(f"[warn] {len(not_found)} problems not found anywhere:")
        for t in sorted(not_found):
            print(f"  {t}")


if __name__ == "__main__":
    main()
