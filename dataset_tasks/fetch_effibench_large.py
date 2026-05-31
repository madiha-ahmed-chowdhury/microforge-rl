"""
fetch_effibench_large.py — Sample 30 EffiBench problems with HIGH algorithmic complexity.

Strategy:
  - Filter for HIGH stress_level problems first (DP, graph, tree, dijkstra, etc.)
    These are computationally hard regardless of input size — O(n²), O(n³), exponential.
  - Among those, pick test cases that fit COMPLETELY within MAX_INPUT chars (no truncation).
  - Pick the LARGEST fitting test case per problem (most compute stress, still complete).

Why this beats filtering by input size:
  - A summation problem with n=10^9 has a huge input but trivial O(n) compute.
  - A graph/DP problem with n=200 has a tiny input but hard O(n²)+ compute.
  - We want CPU stress, not I/O stress — so complexity keyword > input size.

Target:
  - MAX_INPUT = 5000 chars — fits completely in JSON, no truncation, correctness testable.
  - Priority: high-stress > medium-stress, then by largest fitting input within each tier.

Saves to dataset_tasks/effibench_problems_large.json
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

random.seed(42)

N_SAMPLE  = 30
MIN_INPUT = 50       # ignore trivial empty/single-token inputs
MAX_INPUT = 5_000    # 5KB max — guaranteed complete, no truncation ever
OUT_PATH  = Path(__file__).parent / "effibench_problems_large.json"

HIGH_KW = [
    "dynamic programming", "dp", "graph", "tree", "dijkstra", "floyd",
    "bellman", "shortest path", "minimum spanning", "topological",
    "strongly connected", "dfs", "bfs", "permutation", "backtrack",
    "segment tree", "fenwick", "binary indexed", "matrix", "n^2", "n^3",
    "n²", "n³", "combinatorics", "memoization", "knapsack", "subset sum",
    "traveling salesman", "tsp", "hamiltonian", "clique", "independent set",
    "10^5", "10^4", "100000", "10000", "heap", "priority queue",
    "binary search", "divide and conquer", "merge sort", "quick sort",
]
MED_KW = [
    "sort", "search", "hash", "string", "recursion", "palindrome",
    "queue", "stack", "greedy", "two pointer", "sliding window",
]


def stress_level(desc: str) -> str:
    d = desc.lower()
    if any(k in d for k in HIGH_KW):
        return "high"
    if any(k in d for k in MED_KW):
        return "medium"
    return "low"


def get_python_solution(solutions: dict) -> str | None:
    if not solutions or not isinstance(solutions, dict):
        return None
    for key in ("python3", "python", "Python3", "Python"):
        sol = solutions.get(key)
        if sol and isinstance(sol, dict):
            return sol.get("code", "")
        if sol and isinstance(sol, str):
            return sol
    return None


def parse_all_tests(row: dict) -> list[dict]:
    raw = row.get("generated_tests") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    cases = []
    for tc in raw:
        if isinstance(tc, dict):
            cases.append({
                "input":  str(tc.get("input",  "")),
                "output": str(tc.get("output", "")),
            })
    return cases


def pick_best_cases(all_cases: list[dict], n: int = 3) -> list[dict]:
    """Return the n largest complete test cases (fitting within MAX_INPUT)."""
    fitting = [tc for tc in all_cases if MIN_INPUT <= len(tc["input"]) <= MAX_INPUT]
    fitting.sort(key=lambda tc: len(tc["input"]), reverse=True)
    return fitting[:n]


def build_prompt(description: str, test_cases: list[dict]) -> str:
    prompt = (
        description.strip()
        + "\n\nWrite a complete Python script that reads from stdin and writes to stdout. "
        "Return only the code, no explanation, no markdown fences."
    )
    if test_cases:
        tc = test_cases[-1]   # smallest of selected cases for the example
        prompt += f"\n\nExample input:\n{tc['input'][:500]}"
        prompt += f"\nExpected output:\n{tc['output'][:200]}"
    return prompt


def main():
    print("[fetch_large] Loading EffiBench/effibench-x (test split)...")
    ds = load_dataset("EffiBench/effibench-x", split="test")
    print(f"[fetch_large] Loaded {len(ds)} problems")

    high_candidates = []
    med_candidates  = []

    for row in ds:
        py_sol = get_python_solution(row.get("solutions") or {})
        if not py_sol:
            continue
        description = (row.get("description") or row.get("description_md") or "").strip()
        if not description:
            continue
        all_cases  = parse_all_tests(row)
        best_cases = pick_best_cases(all_cases, n=3)
        if not best_cases:
            continue
        sl         = stress_level(description)
        max_size   = max(len(tc["input"]) for tc in best_cases)
        entry      = (max_size, row, best_cases)
        if sl == "high":
            high_candidates.append(entry)
        elif sl == "medium":
            med_candidates.append(entry)

    # sort each tier by largest fitting input (more data = more compute work)
    high_candidates.sort(key=lambda x: x[0], reverse=True)
    med_candidates.sort(key=lambda x: x[0], reverse=True)

    print(f"[fetch_large] High-complexity candidates : {len(high_candidates)}")
    print(f"[fetch_large] Medium-complexity candidates: {len(med_candidates)}")
    if high_candidates:
        print(f"[fetch_large] High input sizes (top 5): {[s[0] for s in high_candidates[:5]]}")

    # take from high first, fill remainder from medium
    pool = high_candidates[:N_SAMPLE * 2] + med_candidates[:N_SAMPLE]
    random.shuffle(pool)
    selected = pool[:N_SAMPLE]
    if len(selected) < N_SAMPLE:
        print(f"[fetch_large] WARNING: only {len(selected)} problems found")

    problems = []
    for i, (max_size, row, best_cases) in enumerate(selected):
        description   = (row.get("description") or row.get("description_md") or "").strip()
        py_solution   = get_python_solution(row.get("solutions") or {})
        sl            = stress_level(description)
        time_limit_ms = (row.get("time_limit_nanos") or 0) // 1_000_000
        mem_limit_mb  = (row.get("memory_limit_bytes") or 0) // (1024 * 1024)

        # Full inputs — guaranteed ≤ MAX_INPUT so no truncation
        full_cases = [
            {"input": tc["input"], "output": tc["output"]}
            for tc in best_cases
        ]

        problems.append({
            "problem_id":         f"effi_large_{i:04d}",
            "source":             "EffiBench",
            "task_type":          3,
            "stress_level":       sl,
            "description":        description,
            "prompt":             build_prompt(description, best_cases),
            "test_cases":         full_cases,
            "max_input_size":     max_size,
            "total_test_cases":   len(parse_all_tests(row)),
            "canonical_solution": py_solution,
            "has_generator":      bool(row.get("test_case_generator")),
            "time_limit_ms":      time_limit_ms,
            "memory_limit_mb":    mem_limit_mb,
            "difficulty":         row.get("difficulty") or "",
            "source_url":         row.get("url") or "",
            "tags":               row.get("tags") or [],
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(problems, f, indent=2)

    stress_counts = {"high": 0, "medium": 0, "low": 0}
    for p in problems:
        stress_counts[p["stress_level"]] += 1

    input_sizes = [p["max_input_size"] for p in problems]
    json_size   = OUT_PATH.stat().st_size

    print(f"\n[fetch_large] Saved {len(problems)} problems → {OUT_PATH}")
    print(f"  JSON file size  : {json_size / 1024:.1f} KB")
    if input_sizes:
        print(f"  Input size range: min={min(input_sizes)}, max={max(input_sizes)}, avg={sum(input_sizes)/len(input_sizes):.0f} chars")
    print(f"  Stress          : high={stress_counts['high']}, medium={stress_counts['medium']}, low={stress_counts['low']}")
    print(f"  Has canonical   : {sum(1 for p in problems if p['canonical_solution'])}/{len(problems)}")


if __name__ == "__main__":
    main()
