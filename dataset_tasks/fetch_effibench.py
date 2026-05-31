"""
fetch_effibench.py — Sample problems from EffiBench/effibench-x.

623 competitive programming problems, each with:
  - 100 pre-generated test cases (generated_tests)
  - test_case_generator: Python code to produce large stress inputs
  - canonical solutions in multiple languages
  - time/memory limits from the original judge

Saves to dataset_tasks/effibench_problems.json
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

random.seed(42)

N_SAMPLE = 100  # number of problems to sample
OUT_PATH = Path(__file__).parent / "effibench_problems.json"

HIGH_KW = ["10^6","10^5","1000000","100000","graph","tree","dynamic programming",
           "dp","matrix","dijkstra","permutation","backtrack","segment tree",
           "binary search","1e6","1e5","n^2","n^3","heap","shortest path",
           "minimum spanning","topological","strongly connected"]
MED_KW  = ["sort","search","hash","string","recursion","palindrome","queue","stack",
           "greedy","priority","binary","two pointer"]


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


def extract_test_cases(row: dict, max_cases: int = 3) -> list[dict]:
    raw = row.get("generated_tests") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []
    cases = []
    for tc in raw[:max_cases]:
        if isinstance(tc, dict):
            cases.append({
                "input":  str(tc.get("input",  "")),
                "output": str(tc.get("output", "")),
            })
    return cases


def get_stress_input(row: dict) -> str:
    """Return the largest available test input from generated_tests."""
    raw = row.get("generated_tests") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return ""
    if not raw:
        return ""
    # pick the test case with the largest input (likely most complex)
    biggest = max(raw, key=lambda tc: len(str(tc.get("input", ""))))
    return str(biggest.get("input", ""))


def build_prompt(description: str, test_cases: list[dict]) -> str:
    prompt = (
        description.strip()
        + "\n\nWrite a complete Python script that reads from stdin and writes to stdout. "
        "Return only the code, no explanation, no markdown fences."
    )
    if test_cases:
        tc = test_cases[0]
        prompt += f"\n\nExample input:\n{tc['input'][:300]}"
        prompt += f"\nExpected output:\n{tc['output'][:200]}"
    return prompt


def main():
    print("[fetch_effibench] Loading EffiBench/effibench-x (test split)...")
    ds = load_dataset("EffiBench/effibench-x", split="test")
    print(f"[fetch_effibench] Loaded {len(ds)} problems")

    rows = list(ds)
    random.shuffle(rows)

    # filter to problems that have a Python3 solution
    python_rows = [r for r in rows if get_python_solution(r.get("solutions") or {})]
    print(f"[fetch_effibench] {len(python_rows)} problems have a Python3 solution")

    problems = []
    for i, row in enumerate(python_rows):
        if len(problems) >= N_SAMPLE:
            break

        description = (row.get("description") or row.get("description_md") or "").strip()
        if not description:
            continue

        test_cases  = extract_test_cases(row)
        stress_inp  = get_stress_input(row)
        py_solution = get_python_solution(row.get("solutions") or {})
        sl          = stress_level(description)

        # time/memory limits from original judge
        time_limit_ms   = (row.get("time_limit_nanos") or 0) // 1_000_000
        memory_limit_mb = (row.get("memory_limit_bytes") or 0) // (1024 * 1024)

        problem = {
            "problem_id":         f"effi_{i:04d}",
            "source":             "EffiBench",
            "task_type":          3,
            "stress_level":       sl,
            "description":        description,
            "prompt":             build_prompt(description, test_cases),
            "test_cases":         test_cases,
            "stress_input":       stress_inp,
            "canonical_solution": py_solution,
            "has_generator":      bool(row.get("test_case_generator")),
            "time_limit_ms":      time_limit_ms,
            "memory_limit_mb":    memory_limit_mb,
            "difficulty":         row.get("difficulty") or "",
            "source_url":         row.get("url") or "",
            "tags":               row.get("tags") or [],
        }
        problems.append(problem)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(problems, f, indent=2)

    # Summary
    stress_counts = {"high": 0, "medium": 0, "low": 0}
    for p in problems:
        stress_counts[p["stress_level"]] += 1

    has_stress_input   = sum(1 for p in problems if p["stress_input"])
    has_canonical      = sum(1 for p in problems if p["canonical_solution"])
    has_generator      = sum(1 for p in problems if p["has_generator"])
    tc_counts          = [len(p["test_cases"]) for p in problems]

    print(f"\n[fetch_effibench] Saved {len(problems)} problems → {OUT_PATH}")
    print(f"  Stress           : high={stress_counts['high']}, medium={stress_counts['medium']}, low={stress_counts['low']}")
    print(f"  Has stress_input : {has_stress_input}/{len(problems)}")
    print(f"  Has canonical    : {has_canonical}/{len(problems)}")
    print(f"  Has generator    : {has_generator}/{len(problems)}")
    print(f"  Test cases       : min={min(tc_counts)}, max={max(tc_counts)}, avg={sum(tc_counts)/len(tc_counts):.1f}")


if __name__ == "__main__":
    main()
