#!/usr/bin/env python3
"""
test_with_generator.py — Test one hard EffiBench problem across all 14 action configs.

Flow:
  1. [Host]     Fetch a hard problem with a generator from EffiBench (HuggingFace cache)
  2. [Host]     Call LLM → solution code
  3. [VM once]  Run test_case_generator → stress input (reused for ALL configs)
  4. [VM once]  Run canonical solution with that input → expected output (oracle)
  5. [Per config, fresh VM]:
       - Apply cgroups (cpu/mem limits)
       - Run LLM solution with stress input
       - Compare output to expected → tests_passed
       - Record transition (state, action, reward, next_state)
  6. [Host]     Print summary of all 14 transitions

Usage:
    venv/bin/python test_with_generator.py
    venv/bin/python test_with_generator.py --idx 2
"""

import argparse
import datetime
import json
import time
import uuid
from datasets import load_dataset

from actions import ACTION_CONFIGS, build_state, compute_reward, sha8, static_analyse
from llm import CLAUDE_MODEL, MINIMAX_MODEL, build_llm_prompt, call_minimax, strip_code
from vm import apply_cgroups, send_code, start_vm, stop_vm, wait_for_agent

# ── Config ────────────────────────────────────────────────────────────────────

HIGH_KW = [
    "dynamic programming", "dp", "graph", "tree", "dijkstra", "shortest path",
    "segment tree", "knapsack", "permutation", "backtrack", "topological",
    "strongly connected", "matrix", "binary search", "heap",
]

GENERATOR_TIMEOUT_MS = 15_000   # generator can be slow
OUTPUT_REPORT = "test_gen_report.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def find_hard_problems_with_generator(ds, max_results: int = 10) -> list[dict]:
    """Return problems that are high-complexity AND have a test_case_generator."""
    found = []
    for row in ds:
        desc = (row.get("description") or "").lower()
        if not any(k in desc for k in HIGH_KW):
            continue
        if not row.get("test_case_generator"):
            continue
        if not get_python_solution(row.get("solutions") or {}):
            continue
        found.append(row)
        if len(found) >= max_results:
            break
    return found


def run_generator_in_vm(vm, generator_code: str) -> str | None:
    """
    Run the test_case_generator inside the VM.
    The generator defines generate_test_cases(num_cases, seed) and returns a list of dicts.
    We call it with num_cases=1 and extract the input of the first case.
    Returns the generated input string, or None on failure.
    """
    script = f"""
import json as _json
import sys as _sys

{generator_code}

try:
    cases = generate_test_cases(num_cases=1, seed=42)
    if cases and isinstance(cases, list):
        tc = cases[0]
        inp = tc.get('input', '') if isinstance(tc, dict) else ''
        print(str(inp), end='')
    else:
        print('', end='')
except Exception as e:
    _sys.stderr.write(f'GENERATOR ERROR: {{e}}\\n')
    print('', end='')
"""
    result = send_code(vm, script, GENERATOR_TIMEOUT_MS)
    if result["exit_code"] != 0 or result.get("stderr"):
        print(f"[gen] stderr: {result.get('stderr', '')[:300]}")
    stdin_data = result.get("stdout", "").strip()
    return stdin_data if stdin_data else None


def run_solution_in_vm(vm, code: str, stdin_data: str, timeout_ms: int) -> dict:
    """Run a stdin/stdout solution with the given input."""
    stdin_mock = (
        "import sys as _sys, io as _io\n"
        f"_sys.stdin = _io.StringIO({repr(stdin_data)})\n"
    )
    return send_code(vm, stdin_mock + code, timeout_ms)


# ── Main ──────────────────────────────────────────────────────────────────────

def boot_vm(vm_id: str) -> object | None:
    """Boot a VM and wait for the agent. Returns VMState or None on failure."""
    vm = start_vm(vm_id)
    time.sleep(3)
    if not wait_for_agent(vm):
        stop_vm(vm)
        return None
    return vm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0,
                        help="Index into the filtered hard-problem list (default: 0)")
    args = parser.parse_args()

    # ── 1. Load dataset and pick problem ──────────────────────────────────────
    print("[test] Loading EffiBench from HuggingFace cache...")
    ds = load_dataset("EffiBench/effibench-x", split="test")
    problems = find_hard_problems_with_generator(ds, max_results=20)
    print(f"[test] Found {len(problems)} hard problems with generators")

    if args.idx >= len(problems):
        print(f"[test] ERROR: idx={args.idx} out of range (max {len(problems)-1})")
        return

    row            = problems[args.idx]
    description    = (row.get("description") or row.get("description_md") or "").strip()
    generator_code = row.get("test_case_generator", "")
    canonical_code = get_python_solution(row.get("solutions") or {})
    url            = row.get("url", "")

    print(f"\n[test] Problem : {url}")
    print(f"[test] Generator: {len(generator_code)} chars")
    print(f"[test] Canonical: {len(canonical_code)} chars")
    print(f"[test] Description (first 300):\n{description[:300]}\n")

    # ── 2. Call LLM on host ───────────────────────────────────────────────────
    instance = {
        "dataset": "effibench", "task_id": "test_gen",
        "prompt": description, "task_type": 3, "test_cases": [],
    }
    llm_prompt, expected_func, prefill = build_llm_prompt(instance)
    print(f"[test] Calling LLM ({MINIMAX_MODEL})...")
    llm_start = time.monotonic()
    raw_text, ptok, ctok = call_minimax(llm_prompt, expected_func=expected_func, prefill=prefill, stress_level="high")
    llm_ms = int((time.monotonic() - llm_start) * 1000)
    generated_code = strip_code(raw_text)
    print(f"[test] LLM done in {llm_ms}ms | {len(generated_code)} chars")
    print("─" * 60)
    print(generated_code[:600], "..." if len(generated_code) > 600 else "")
    print("─" * 60)

    code_features = static_analyse(generated_code)
    state         = build_state(description, code_features, 3)

    # ── 3. Generate stress input ONCE (reused for all 14 configs) ─────────────
    # Use the highest-resource config for generation + canonical (max CPU/mem = best chance of success)
    highest = ACTION_CONFIGS[-1]  # 500mc / 256MB / 10s
    print(f"\n[test] Booting prep VM (highest config: {highest['cpu_millicores']}mc/{highest['memory_limit_mb']}MB)...")
    prep_vm = boot_vm("prep-gen-vm")
    if not prep_vm:
        print("[test] ERROR: prep VM failed to start")
        return
    apply_cgroups(prep_vm, highest["cpu_millicores"], highest["memory_limit_mb"])

    print("[test] Running generator in VM...")
    stdin_data = run_generator_in_vm(prep_vm, generator_code)
    if not stdin_data:
        print("[test] ERROR: generator produced no output")
        stop_vm(prep_vm)
        return
    print(f"[test] Generated input: {len(stdin_data)} chars")
    print(f"[test] Preview: {repr(stdin_data[:200])}")

    # ── 4. Run canonical solution once to get expected output (oracle) ─────────
    print("\n[test] Running canonical solution (oracle)...")
    ref_exec   = run_solution_in_vm(prep_vm, canonical_code, stdin_data, 15_000)
    ref_output = ref_exec.get("stdout", "").strip()
    ref_ok     = ref_exec["exit_code"] == 0 and bool(ref_output)
    print(f"[test] Canonical: exit={ref_exec['exit_code']} | output={len(ref_output)} chars")
    if not ref_ok:
        print(f"[test] WARNING: canonical failed — tests_passed will be None for all configs")
        print(f"[test] stderr: {ref_exec.get('stderr','')[:300]}")
    stop_vm(prep_vm)

    # ── 5. Run LLM solution across all 14 action configs (fresh VM each) ───────
    print(f"\n[test] Running LLM solution across {len(ACTION_CONFIGS)} configs (fresh VM each)...")
    transitions = []

    for action_idx, action in enumerate(ACTION_CONFIGS):
        vm_id      = f"run-{uuid.uuid4().hex[:6]}"
        current_vm = boot_vm(vm_id)
        if not current_vm:
            print(f"[test] WARNING: VM failed for config {action_idx}, skipping")
            continue
        apply_cgroups(current_vm, action["cpu_millicores"], action["memory_limit_mb"])

        try:
            execution = run_solution_in_vm(current_vm, generated_code, stdin_data, action["timeout_ms"])
        except Exception as e:
            execution = {
                "exit_code": -1, "timed_out": False, "wall_time_ms": 0,
                "cpu_user_ms": 0, "cpu_sys_ms": 0, "mem_peak_kb": 0,
                "stdout": "", "stderr": str(e),
            }

        if ref_ok and execution["exit_code"] == 0:
            llm_output   = execution.get("stdout", "").strip()
            tests_passed = (llm_output == ref_output)
        elif not ref_ok:
            tests_passed = None
        else:
            tests_passed = False

        stop_vm(current_vm)

        reward_dict = compute_reward(execution, action, tests_passed)
        next_state  = dict(state)
        success     = 1 if execution["exit_code"] == 0 else 0
        next_state["recent_success_rate"]  = round(success * 0.1 + state["recent_success_rate"] * 0.9, 4)
        next_state["recent_mean_cpu_used"] = round(execution["cpu_user_ms"] * 0.1 + state["recent_mean_cpu_used"] * 0.9, 2)
        next_state["recent_mean_mem_used"] = round(execution["mem_peak_kb"] * 0.1 + state["recent_mean_mem_used"] * 0.9, 2)

        status = "TIMEOUT" if execution["timed_out"] else ("OK" if execution["exit_code"] == 0 else f"ERR({execution['exit_code']})")
        tp     = "T" if tests_passed is True else ("F" if tests_passed is False else "N")
        print(f"  [{action_idx+1:2d}/{len(ACTION_CONFIGS)}] "
              f"{action['cpu_millicores']:3d}mc/{action['memory_limit_mb']:3d}MB/{action['timeout_ms']:5d}ms  "
              f"{status:10s}  wall={execution['wall_time_ms']:5d}ms  tp={tp}  r={reward_dict['r']:+.3f}")

        transitions.append({
            "episode_id": f"ep_{int(time.time())}_test_gen_{action_idx}",
            "task_id":    "test_gen",
            "dataset":    "effibench",
            "step":       0,
            "done":       True,
            "state":      state,
            "action": {
                "cpu_millicores":  action["cpu_millicores"],
                "memory_limit_mb": action["memory_limit_mb"],
                "timeout_ms":      action["timeout_ms"],
            },
            "execution": {
                "exit_code":    execution["exit_code"],
                "timed_out":    execution["timed_out"],
                "wall_time_ms": execution["wall_time_ms"],
                "cpu_user_ms":  execution["cpu_user_ms"],
                "cpu_sys_ms":   execution["cpu_sys_ms"],
                "mem_peak_kb":  execution["mem_peak_kb"],
                "tests_passed": tests_passed,
            },
            "reward":            reward_dict["r"],
            "reward_components": {k: v for k, v in reward_dict.items() if k != "r"},
            "next_state":        next_state,
            "meta": {
                "code_hash":    sha8(generated_code),
                "llm_model":    CLAUDE_MODEL,
                "collected_at": datetime.datetime.utcnow().isoformat() + "Z",
            },
        })

    # ── 6. Print detailed per-VM results ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("GENERATED CODE (Claude):")
    print("─" * 60)
    print(generated_code)
    print("─" * 60)

    print(f"\nGENERATED TEST INPUT ({len(stdin_data)} chars, showing first 500):")
    print("─" * 60)
    print(stdin_data[:500], "...(truncated)" if len(stdin_data) > 500 else "")
    print("─" * 60)

    print(f"\nEXPECTED OUTPUT (canonical, {len(ref_output)} chars):")
    print("─" * 60)
    print(ref_output[:500], "...(truncated)" if len(ref_output) > 500 else "")
    print("─" * 60)

    print(f"\nPER-VM RESULTS ({len(transitions)} configs):")
    print(f"{'Config':<30} {'Status':<10} {'Wall(ms)':<10} {'tp':<6} {'Reward'}")
    print("─" * 70)
    vm_results = []
    for t in transitions:
        a  = t["action"]
        ex = t["execution"]
        label  = f"{a['cpu_millicores']}mc/{a['memory_limit_mb']}MB/{a['timeout_ms']}ms"
        status = "TIMEOUT" if ex["timed_out"] else ("OK" if ex["exit_code"] == 0 else f"ERR({ex['exit_code']})")
        tp     = "PASSED" if ex["tests_passed"] is True else ("FAILED" if ex["tests_passed"] is False else "N/A")
        print(f"  {label:<28} {status:<10} {ex['wall_time_ms']:<10} {tp:<6} {t['reward']:+.3f}")
        vm_results.append({
            "config":       label,
            "cpu_millicores":    a["cpu_millicores"],
            "memory_limit_mb":   a["memory_limit_mb"],
            "timeout_ms":        a["timeout_ms"],
            "exit_code":         ex["exit_code"],
            "timed_out":         ex["timed_out"],
            "wall_time_ms":      ex["wall_time_ms"],
            "cpu_user_ms":       ex["cpu_user_ms"],
            "mem_peak_kb":       ex["mem_peak_kb"],
            "tests_passed":      ex["tests_passed"],
            "actual_output":     ex["stdout_snippet"],
            "stderr":            ex["stderr_snippet"],
            "reward":            t["reward"],
        })

    # ── 7. Save full report ────────────────────────────────────────────────────
    report = {
        "problem_url":        url,
        "description":        description,
        "llm_model":          CLAUDE_MODEL,
        "llm_latency_ms":     llm_ms,
        "generated_code":     generated_code,
        "canonical_code":     canonical_code,
        "generated_input_preview": stdin_data[:500] + ("...(truncated)" if len(stdin_data) > 500 else ""),
        "generated_input_size_chars": len(stdin_data),
        "expected_output":    ref_output,
        "canonical_ok":       ref_ok,
        "vm_results":         vm_results,
        "summary": {
            "total_configs":  len(transitions),
            "passed":  sum(1 for r in vm_results if r["tests_passed"] is True),
            "failed":  sum(1 for r in vm_results if r["tests_passed"] is False),
            "na":      sum(1 for r in vm_results if r["tests_passed"] is None),
            "timeouts": sum(1 for r in vm_results if r["timed_out"]),
            "reward_min": min(t["reward"] for t in transitions),
            "reward_max": max(t["reward"] for t in transitions),
        },
        "collected_at": datetime.datetime.utcnow().isoformat() + "Z",
    }

    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    passed = report["summary"]["passed"]
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{len(transitions)} configs passed")
    print(f"Reward range: {report['summary']['reward_min']:+.3f} → {report['summary']['reward_max']:+.3f}")
    print(f"Report saved → {OUTPUT_REPORT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
