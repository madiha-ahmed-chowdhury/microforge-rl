#!/usr/bin/env python3
"""
test_one.py — Test a single dataset instance end-to-end.

Loads one instance (random by default), calls MiniMax, runs the code in a
Firecracker VM, and saves the result to test_transitions.jsonl and
test_code_pairs.jsonl (separate from the main collection output).

Usage:
    venv/bin/python test_one.py                      # random MBPP instance
    venv/bin/python test_one.py --dataset humaneval  # random HumanEval
    venv/bin/python test_one.py --idx 42             # specific instance
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from actions import (
    ACTION_CONFIGS, build_runnable, build_state, compute_reward,
    dataset_size, load_one, sha8, static_analyse,
)
from llm import MINIMAX_MODEL, build_llm_prompt, call_minimax, strip_code
from vm import send_code, start_vm, stop_vm, wait_for_agent

WORK_DIR         = Path(__file__).parent.resolve()
OUTPUT_JSONL     = str(WORK_DIR / "test_transitions.jsonl")
OUTPUT_CODE      = str(WORK_DIR / "test_code_pairs.jsonl")

# Use action config index 1 (500mc / 256MB / 10s) for single tests
ACTION = ACTION_CONFIGS[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mbpp", "humaneval"], default="mbpp")
    parser.add_argument("--idx", type=int, default=None,
                        help="Dataset index (random if not set)")
    args = parser.parse_args()

    idx = args.idx if args.idx is not None else random.randint(0, dataset_size(args.dataset) - 1)
    print(f"\n[test] Loading instance {idx} from {args.dataset}...")
    instance = load_one(args.dataset, idx)
    print(f"[test] task_id={instance['task_id']}")
    print(f"[test] Prompt: {instance['prompt']}")
    if instance.get("test_list"):
        print("[test] Tests:")
        for t in instance["test_list"]:
            print(f"         {t}")

    # ── LLM ───────────────────────────────────────────────────────────────────
    print(f"\n[test] Calling MiniMax ({MINIMAX_MODEL})...")
    llm_start = time.monotonic()
    raw_text, prompt_tokens, completion_tokens = call_minimax(build_llm_prompt(instance))
    llm_latency_ms = int((time.monotonic() - llm_start) * 1000)
    generated_code = strip_code(raw_text)
    print(f"[test] Response in {llm_latency_ms}ms ({prompt_tokens} in / {completion_tokens} out tokens)")
    print("─" * 60)
    print(generated_code[:600], "..." if len(generated_code) > 600 else "")
    print("─" * 60)

    # ── Static analysis ────────────────────────────────────────────────────────
    code_features           = static_analyse(generated_code)
    state                   = build_state(instance["prompt"], code_features, instance["task_type"])
    code_only, code_w_tests = build_runnable(generated_code, instance)
    has_tests               = bool(instance.get("test_list") or instance.get("test_code"))

    # ── Boot VM ────────────────────────────────────────────────────────────────
    vm = start_vm("test-vm")
    time.sleep(3)
    if not wait_for_agent(vm):
        print("[test] ERROR: guest agent did not start.")
        stop_vm(vm)
        return

    # ── Execute ────────────────────────────────────────────────────────────────
    print(f"\n[test] Executing code (timeout={ACTION['timeout_ms']}ms)...")
    try:
        execution = send_code(vm, code_only, ACTION["timeout_ms"])
    except Exception as e:
        execution = {
            "exit_code": -1, "timed_out": False, "wall_time_ms": 0,
            "cpu_user_ms": 0, "cpu_sys_ms": 0, "mem_peak_kb": 0,
            "stdout": "", "stderr": str(e),
        }

    print(f"[test] exit={execution['exit_code']} wall={execution['wall_time_ms']}ms mem={execution['mem_peak_kb']}KB")
    if execution.get("stdout"):
        print(f"[test] stdout: {execution['stdout'][:300]}")
    if execution.get("stderr"):
        print(f"[test] stderr: {execution['stderr'][:300]}")

    # ── Tests ──────────────────────────────────────────────────────────────────
    tests_passed = None
    if has_tests and execution["exit_code"] == 0:
        print("[test] Running tests...")
        try:
            tr = send_code(vm, code_w_tests, ACTION["timeout_ms"])
            tests_passed = (
                tr["exit_code"] == 0 and
                "ALL_TESTS_PASSED" in tr.get("stdout", "")
            )
            print(f"[test] tests_passed={tests_passed}")
            if tr.get("stderr"):
                print(f"[test] test stderr: {tr['stderr'][:300]}")
        except Exception as e:
            print(f"[test] test run error: {e}")
            tests_passed = False

    stop_vm(vm)

    # ── Reward ─────────────────────────────────────────────────────────────────
    reward_dict = compute_reward(execution, ACTION, tests_passed)
    print(f"\n[test] reward={reward_dict['r']} (success={reward_dict['r_success']} correctness={reward_dict['r_correctness']})")

    next_state = dict(state)
    success = 1 if execution["exit_code"] == 0 else 0
    next_state["recent_success_rate"]  = round(success * 0.1 + state["recent_success_rate"] * 0.9, 4)
    next_state["recent_mean_cpu_used"] = round(execution["cpu_user_ms"] * 0.1 + state["recent_mean_cpu_used"] * 0.9, 2)
    next_state["recent_mean_mem_used"] = round(execution["mem_peak_kb"] * 0.1 + state["recent_mean_mem_used"] * 0.9, 2)

    import datetime
    transition = {
        "episode_id":     f"test_{int(time.time())}",
        "step":           0,
        "done":           True,
        "dataset":        instance["dataset"],
        "task_id":        instance["task_id"],
        "generated_code": generated_code,
        "state":          state,
        "action":         ACTION,
        "execution": {
            "exit_code":      execution["exit_code"],
            "timed_out":      execution["timed_out"],
            "wall_time_ms":   execution["wall_time_ms"],
            "cpu_user_ms":    execution["cpu_user_ms"],
            "cpu_sys_ms":     execution["cpu_sys_ms"],
            "mem_peak_kb":    execution["mem_peak_kb"],
            "stdout_snippet": execution.get("stdout", "")[:500],
            "stderr_snippet": execution.get("stderr", "")[:500],
            "tests_passed":   tests_passed,
        },
        "reward":            reward_dict["r"],
        "reward_components": {k: v for k, v in reward_dict.items() if k != "r"},
        "next_state":        next_state,
        "meta": {
            "prompt_hash":       sha8(instance["prompt"]),
            "code_hash":         sha8(generated_code),
            "model":             MINIMAX_MODEL,
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "llm_latency_ms":    llm_latency_ms,
            "collected_at":      datetime.datetime.utcnow().isoformat() + "Z",
            "collector_version": "v1.0-test",
        },
    }

    with open(OUTPUT_JSONL, "a") as f:
        f.write(json.dumps(transition) + "\n")

    code_pair = {
        "task_id":        instance["task_id"],
        "dataset":        instance["dataset"],
        "prompt":         instance["prompt"],
        "generated_code": generated_code,
        "reference_code": instance.get("code", ""),
        "tests_passed":   tests_passed,
        "collected_at":   datetime.datetime.utcnow().isoformat() + "Z",
        "model":          MINIMAX_MODEL,
    }
    with open(OUTPUT_CODE, "a") as f:
        f.write(json.dumps(code_pair) + "\n")

    print(f"\n[test] Saved to {OUTPUT_JSONL}")
    print(f"[test] Saved to {OUTPUT_CODE}")
    print("\n[test] Done.")


if __name__ == "__main__":
    main()
