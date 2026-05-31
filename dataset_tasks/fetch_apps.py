"""
fetch_apps.py — Download and sample 150 problems from the APPS dataset.
Saves to dataset_tasks/apps_problems.json.

Sample: 100 interview + 50 competition problems.
Skips problems with no parseable test cases.
"""

import json
import os
import random
from pathlib import Path

from datasets import load_dataset

random.seed(42)

OUT_PATH = Path(__file__).parent / "apps_problems.json"

HIGH_KW   = ["10^6","10^5","1000000","100000","graph","tree","dynamic programming",
              "dp","matrix","dijkstra","permutation","backtrack","segment tree","binary search"]
MEDIUM_KW = ["sort","search","hash","string","recursion","palindrome"]


def stress_level(difficulty: str, question: str) -> str:
    if difficulty == "competition":
        return "high"
    q = question.lower()
    if any(k in q for k in HIGH_KW):
        return "high"
    if any(k in q for k in MEDIUM_KW):
        return "medium"
    return "low"


def parse_test_cases(row) -> list[dict]:
    raw = row.get("input_output", "") or ""
    if isinstance(raw, dict):
        io = raw
    else:
        try:
            io = json.loads(raw)
        except Exception:
            return []
    inputs  = io.get("inputs",  []) or []
    outputs = io.get("outputs", []) or []
    cases = []
    for inp, out in zip(inputs, outputs):
        # inputs are lists of lines — join them into a proper stdin string
        if isinstance(inp, list):
            inp_str = "\n".join(str(x) for x in inp)
        else:
            inp_str = str(inp)
        if isinstance(out, list):
            out_str = "\n".join(str(x) for x in out)
        else:
            out_str = str(out)
        cases.append({"input": inp_str, "output": out_str})
        if len(cases) >= 3:
            break
    return cases


def build_prompt(question: str, test_cases: list[dict]) -> str:
    prompt = (
        question.strip()
        + "\n\nWrite a complete Python script that reads from stdin and writes to stdout. "
        "Do not use input() without handling EOF. "
        "Return only the code, no explanation, no markdown fences."
    )
    if test_cases:
        tc = test_cases[0]
        prompt += (
            f"\n\nExample input:\n{tc['input'][:300]}"
            f"\nExpected output:\n{tc['output'][:200]}"
        )
    return prompt


def count_test_cases(row) -> int:
    raw = row.get("input_output", "") or ""
    try:
        io = json.loads(raw) if isinstance(raw, str) else raw
        return len(io.get("inputs", []) or [])
    except Exception:
        return 0


def sample_problems(ds, difficulty: str, n: int, min_test_cases: int = 3) -> list[dict]:
    pool = [row for row in ds if row.get("difficulty") == difficulty]
    # sort by number of test cases descending — prefer problems with more test cases
    pool.sort(key=count_test_cases, reverse=True)
    # shuffle within tiers so we don't always pick the same problems
    # tier: >=10, >=5, >=3, >=1
    tier_high   = [r for r in pool if count_test_cases(r) >= 10]
    tier_mid    = [r for r in pool if 5 <= count_test_cases(r) < 10]
    tier_low    = [r for r in pool if min_test_cases <= count_test_cases(r) < 5]
    random.shuffle(tier_high)
    random.shuffle(tier_mid)
    random.shuffle(tier_low)
    pool = tier_high + tier_mid + tier_low

    results = []
    task_type = 1 if difficulty == "interview" else 2
    i = 0
    for row in pool:
        if len(results) >= n:
            break
        test_cases = parse_test_cases(row)
        if len(test_cases) < min_test_cases:
            continue
        question = (row.get("question") or "").strip()
        if not question:
            continue
        sl = stress_level(difficulty, question)
        problem = {
            "problem_id": f"apps_{i:04d}",
            "source":     "APPS",
            "difficulty": difficulty,
            "task_type":  task_type,
            "stress_level": sl,
            "question":   question,
            "prompt":     build_prompt(question, test_cases),
            "test_cases": test_cases,
        }
        url = row.get("url", "")
        if url:
            problem["apps_url"] = url
        results.append(problem)
        i += 1
    return results


def main():
    print("[fetch_apps] Loading APPS dataset from HuggingFace...")
    cache = Path.home() / ".cache/huggingface/hub/datasets--codeparrot--apps/snapshots"
    snapshots = sorted(cache.glob("*/train.jsonl")) if cache.exists() else []
    if snapshots:
        data_file = str(snapshots[-1])
        print(f"[fetch_apps] Using cached file: {data_file}")
    else:
        data_file = "https://huggingface.co/datasets/codeparrot/apps/resolve/main/train.jsonl"
        print("[fetch_apps] Downloading from HuggingFace...")
    ds = load_dataset("json", data_files=data_file, split="train")
    print(f"[fetch_apps] Total rows in dataset: {len(ds)}")

    print("[fetch_apps] Sampling 100 interview problems (prefer ≥3, fallback to ≥1)...")
    interview = sample_problems(ds, "interview", 100, min_test_cases=3)
    if len(interview) < 100:
        print(f"[fetch_apps] Only {len(interview)} with ≥3 test cases, filling remainder with ≥1...")
        existing_ids = {p["problem_id"] for p in interview}
        extra = sample_problems(ds, "interview", 200, min_test_cases=1)
        for p in extra:
            if len(interview) >= 100:
                break
            if p["problem_id"] not in existing_ids:
                interview.append(p)
    print(f"[fetch_apps] Got {len(interview)} interview problems")

    print("[fetch_apps] Sampling 50 competition problems (prefer ≥3, fallback to ≥1)...")
    competition = sample_problems(ds, "competition", 50, min_test_cases=3)
    if len(competition) < 50:
        print(f"[fetch_apps] Only {len(competition)} with ≥3 test cases, filling remainder with ≥1...")
        existing_ids = {p["problem_id"] for p in competition}
        extra = sample_problems(ds, "competition", 100, min_test_cases=1)
        for p in extra:
            if len(competition) >= 50:
                break
            if p["problem_id"] not in existing_ids:
                competition.append(p)
    print(f"[fetch_apps] Got {len(competition)} competition problems")

    problems = interview + competition
    # re-index ids sequentially
    for idx, p in enumerate(problems):
        p["problem_id"] = f"apps_{idx:04d}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(problems, f, indent=2)

    # Summary
    stress_counts = {"high": 0, "medium": 0, "low": 0}
    diff_counts   = {"interview": 0, "competition": 0}
    for p in problems:
        stress_counts[p["stress_level"]] += 1
        diff_counts[p["difficulty"]]     += 1

    tc_counts = [len(p["test_cases"]) for p in problems]
    print(f"\n[fetch_apps] Saved {len(problems)} problems → {OUT_PATH}")
    print(f"  Difficulty  : interview={diff_counts['interview']}, competition={diff_counts['competition']}")
    print(f"  Stress      : high={stress_counts['high']}, medium={stress_counts['medium']}, low={stress_counts['low']}")
    print(f"  Test cases  : min={min(tc_counts)}, max={max(tc_counts)}, avg={sum(tc_counts)/len(tc_counts):.1f}")


if __name__ == "__main__":
    main()
