#!/usr/bin/env python3
"""
collect_dataset.py — Collect RL transitions from MBPP/HumanEval datasets.

For each dataset instance:
  1. Call Ollama to generate code
  2. Run the code in Firecracker microVM with 4 different resource allocations
  3. If test cases exist, run them too and factor into reward
  4. Save all transitions to dataset_transitions.jsonl

The VM is booted once and reused across all episodes for speed.

Usage:
    python3 collect_dataset.py --dataset mbpp --n 3
    python3 collect_dataset.py --dataset humaneval --n 3
"""

import os
import sys
import json
import math
import time
import socket
import hashlib
import subprocess
import datetime
import ast
import re
import argparse
import urllib.request
import urllib.error
from pathlib import Path

import pyarrow.ipc as ipc

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

WORK_DIR        = Path(__file__).parent.resolve()
FIRECRACKER_BIN = str(WORK_DIR / "firecracker")
VM_CONFIG       = str(WORK_DIR / "vm_config.json")
VSOCK_SOCK      = str(WORK_DIR / "vsock.sock")
OUTPUT_JSONL    = str(WORK_DIR / "dataset_transitions.jsonl")

OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "qwen2.5-coder:3b"
VSOCK_PORT      = 52
AGENT_BOOT_TIMEOUT = 180

MBPP_PATH = str(WORK_DIR / "datasets/google-research-datasets___mbpp/full/0.0.0/4bb6404fdc6cacfda99d4ac4205087b89d32030c/mbpp-test.arrow")
HUMANEVAL_PATH = str(WORK_DIR / "datasets/openai_humaneval/openai_humaneval/0.0.0/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval-test.arrow")

# 4 resource allocation configs to try per instance
ACTION_CONFIGS = [
    {"cpu_millicores": 250,  "memory_limit_mb": 128, "timeout_ms": 5000,  "cpu_idx": 0, "mem_idx": 0, "timeout_idx": 0},
    {"cpu_millicores": 500,  "memory_limit_mb": 256, "timeout_ms": 10000, "cpu_idx": 1, "mem_idx": 1, "timeout_idx": 1},
    {"cpu_millicores": 1000, "memory_limit_mb": 256, "timeout_ms": 10000, "cpu_idx": 2, "mem_idx": 1, "timeout_idx": 1},
    {"cpu_millicores": 1000, "memory_limit_mb": 256, "timeout_ms": 30000, "cpu_idx": 2, "mem_idx": 1, "timeout_idx": 2},
]

# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_mbpp(n: int) -> list:
    with open(MBPP_PATH, "rb") as f:
        table = ipc.open_stream(f).read_all()
    samples = []
    for i in range(min(n, table.num_rows)):
        row = {col: table[col][i].as_py() for col in table.schema.names}
        samples.append({
            "dataset":    "mbpp",
            "task_id":    str(row["task_id"]),
            "prompt":     row["text"],
            "test_list":  row["test_list"],      # list of assert strings
            "task_type":  1,
        })
    return samples


def load_humaneval(n: int) -> list:
    with open(HUMANEVAL_PATH, "rb") as f:
        table = ipc.open_stream(f).read_all()
    samples = []
    for i in range(min(n, table.num_rows)):
        row = {col: table[col][i].as_py() for col in table.schema.names}
        samples.append({
            "dataset":      "humaneval",
            "task_id":      row["task_id"],
            "prompt":       row["prompt"],
            "test_code":    row["test"],          # check() function string
            "entry_point":  row["entry_point"],
            "task_type":    2,
        })
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# CODE + TEST ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_runnable(generated_code: str, instance: dict) -> tuple[str, str]:
    """
    Returns (code_only, code_with_tests).
    code_with_tests appends assertions so we can check correctness.
    """
    if instance["dataset"] == "mbpp":
        tests = "\n".join(instance.get("test_list", []))
        code_with_tests = generated_code + "\n\n# --- tests ---\n" + tests + "\nprint('ALL_TESTS_PASSED')\n"
        return generated_code, code_with_tests

    elif instance["dataset"] == "humaneval":
        test_code = instance.get("test_code", "")
        entry    = instance.get("entry_point", "")
        code_with_tests = (
            generated_code + "\n\n"
            + test_code + "\n\n"
            + f"check({entry})\n"
            + "print('ALL_TESTS_PASSED')\n"
        )
        return generated_code, code_with_tests

    return generated_code, generated_code


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS (mirrors run_host.py)
# ─────────────────────────────────────────────────────────────────────────────

def sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def count_prompt_tokens_approx(text: str) -> int:
    return len(text) // 4


def call_ollama(prompt: str, timeout_sec: int = 120) -> dict:
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read())


def strip_code(text: str) -> str:
    fenced = re.search(r'```(?:python)?\n?(.*?)```', text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    clean = []
    for line in text.splitlines():
        s = line.strip()
        is_prose = (s and s[0].isupper() and '=' not in s and '(' not in s and ':' not in s and not s.startswith('#'))
        if not is_prose:
            clean.append(line)
    return '\n'.join(clean).strip()


def static_analyse(code: str) -> dict:
    features = {"line_count": 0, "cyclomatic_complexity": 1, "ast_node_count": 0,
                "has_recursion": 0, "has_external_calls": 0, "max_loop_depth": 0, "estimated_complexity": 3}
    lines = code.strip().splitlines()
    features["line_count"] = len(lines)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return features
    features["ast_node_count"] = sum(1 for _ in ast.walk(tree))
    branch_types = (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With, ast.Assert,
                    ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)
    features["cyclomatic_complexity"] = 1 + sum(1 for n in ast.walk(tree) if isinstance(n, branch_types))
    def max_loop_depth(node, depth=0):
        if isinstance(node, (ast.For, ast.While)):
            depth += 1
        return max([depth] + [max_loop_depth(c, depth) for c in ast.iter_child_nodes(node)])
    features["max_loop_depth"] = max_loop_depth(tree)
    defined_funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    called_names  = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    features["has_recursion"] = int(bool(defined_funcs & called_names))
    external_indicators = {"subprocess", "requests", "urllib", "open", "socket", "http"}
    features["has_external_calls"] = int(any(ind in code.lower() for ind in external_indicators))
    if features["max_loop_depth"] == 0:   features["estimated_complexity"] = 0
    elif features["max_loop_depth"] == 1: features["estimated_complexity"] = 1
    elif features["max_loop_depth"] == 2: features["estimated_complexity"] = 2
    else:                                 features["estimated_complexity"] = 3
    return features


def build_state(prompt: str, code_features: dict, task_type: int) -> dict:
    return {
        "prompt_token_count":      count_prompt_tokens_approx(prompt),
        "prompt_complexity_score": round(len(set(prompt.split())) / max(len(prompt.split()), 1), 3),
        "task_type":               task_type,
        "has_loops_hint":          int("loop" in prompt.lower() or "repeat" in prompt.lower()),
        "has_io_hint":             int("file" in prompt.lower() or "read" in prompt.lower()),
        "example_count":           0,
        **code_features,
        "host_cpu_load_1m":        0.0,
        "host_mem_available_mb":   512,
        "queue_depth":             0,
        "recent_success_rate":     1.0,
        "recent_mean_cpu_used":    0.0,
        "recent_mean_mem_used":    0.0,
    }


def compute_reward(execution: dict, action: dict, tests_passed: bool | None) -> dict:
    r_success  = 1.0 if execution["exit_code"] == 0 else -2.0
    r_timeout  = -5.0 if execution["timed_out"] else 0.0

    wall_sec      = max(execution["wall_time_ms"] / 1000, 0.001)
    cpu_budget_ms = (action["cpu_millicores"] / 1000) * wall_sec * 1000
    cpu_used_ms   = execution["cpu_user_ms"] + execution["cpu_sys_ms"]
    cpu_waste     = max(0, (cpu_budget_ms - cpu_used_ms) / max(cpu_budget_ms, 1))
    mem_alloc_kb  = action["memory_limit_mb"] * 1024
    mem_used_kb   = execution["mem_peak_kb"]
    mem_waste     = max(0, (mem_alloc_kb - mem_used_kb) / max(mem_alloc_kb, 1))

    r_resource = -0.3 * cpu_waste - 0.3 * mem_waste
    r_latency  = -0.2 * math.log(execution["wall_time_ms"] / 500 + 1)

    # test correctness bonus/penalty (only when tests were run)
    if tests_passed is True:
        r_correctness = 1.0
    elif tests_passed is False:
        r_correctness = -1.0
    else:
        r_correctness = 0.0

    total = r_success + r_timeout + r_resource + r_latency + r_correctness
    return {
        "r":             round(total, 4),
        "r_success":     round(r_success, 4),
        "r_timeout":     round(r_timeout, 4),
        "r_resource":    round(r_resource, 4),
        "r_latency":     round(r_latency, 4),
        "r_correctness": round(r_correctness, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIRECRACKER
# ─────────────────────────────────────────────────────────────────────────────

def start_firecracker() -> subprocess.Popen:
    for f in [VSOCK_SOCK, str(WORK_DIR / "firecracker.socket")]:
        if os.path.exists(f):
            os.remove(f)
    cmd = [
        FIRECRACKER_BIN,
        "--api-sock", str(WORK_DIR / "firecracker.socket"),
        "--config-file", VM_CONFIG,
        "--log-path", str(WORK_DIR / "firecracker.log"),
        "--level", "Info",
    ]
    print("[host] Starting Firecracker...")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)


def _vsock_connect(timeout_sec: float) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_sec)
    sock.connect(VSOCK_SOCK)
    sock.sendall(f"CONNECT {VSOCK_PORT}\n".encode())
    resp = b""
    while b"\n" not in resp:
        chunk = sock.recv(32)
        if not chunk:
            break
        resp += chunk
    if not resp.startswith(b"OK "):
        sock.close()
        raise OSError(f"vsock handshake failed: {resp!r}")
    return sock


def wait_for_agent(timeout_sec: int = AGENT_BOOT_TIMEOUT) -> bool:
    print(f"[host] Waiting up to {timeout_sec}s for guest agent...")
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            sock = _vsock_connect(2)
            sock.sendall(json.dumps({"code": "print('ping')", "timeout": 3}).encode() + b"\n")
            response = b""
            while b"\n" not in response:
                chunk = sock.recv(1024)
                if not chunk:
                    break
                response += chunk
            sock.close()
            if json.loads(response.decode()).get("exit_code") == 0:
                print("[host] Guest agent ready.")
                return True
        except (ConnectionRefusedError, OSError, json.JSONDecodeError, socket.timeout):
            time.sleep(1)
    return False


def send_code_to_guest(code: str, timeout_ms: int) -> dict:
    sock = _vsock_connect(timeout_ms / 1000 + 15)
    sock.settimeout(timeout_ms / 1000 + 15)
    sock.sendall((json.dumps({"code": code, "timeout": timeout_ms // 1000}) + "\n").encode("utf-8"))
    response = b""
    while b"\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            break
        response += chunk
    sock.close()
    return json.loads(response.decode("utf-8"))


def stop_firecracker(proc: subprocess.Popen):
    print("[host] Stopping Firecracker...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("[host] Firecracker stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mbpp", "humaneval"], default="mbpp")
    parser.add_argument("--n", type=int, default=3, help="Number of dataset instances to run")
    args = parser.parse_args()

    # Load dataset
    print(f"\n[host] Loading {args.n} instances from {args.dataset}...")
    if args.dataset == "mbpp":
        instances = load_mbpp(args.n)
    else:
        instances = load_humaneval(args.n)
    print(f"[host] Loaded {len(instances)} instances.")

    # Boot VM once — reuse across all episodes
    fc_proc = start_firecracker()
    time.sleep(3)
    if not wait_for_agent():
        print("[host] ERROR: Guest agent did not start.")
        stop_firecracker(fc_proc)
        sys.exit(1)

    transitions_written = 0

    for inst_idx, instance in enumerate(instances):
        print(f"\n{'='*60}")
        print(f"[host] Instance {inst_idx+1}/{len(instances)} | {instance['dataset']} | task_id={instance['task_id']}")
        print(f"[host] Prompt: {instance['prompt'][:120]}...")

        # ── Call Ollama once per instance ──────────────────────────────────
        print(f"[host] Calling Ollama ({OLLAMA_MODEL})...")
        llm_start = time.monotonic()
        try:
            ollama_resp = call_ollama(instance["prompt"])
        except Exception as e:
            print(f"[host] Ollama error: {e}, skipping.")
            continue
        llm_latency_ms = int((time.monotonic() - llm_start) * 1000)
        generated_code = strip_code(ollama_resp["response"])
        print(f"[host] Generated {len(generated_code)} chars in {llm_latency_ms}ms")

        code_features = static_analyse(generated_code)
        code_only, code_with_tests = build_runnable(generated_code, instance)
        prompt_hash = sha8(instance["prompt"])
        code_hash   = sha8(generated_code)

        has_tests = bool(
            instance.get("test_list") or instance.get("test_code")
        )

        # ── Try each action config ─────────────────────────────────────────
        for action_idx, action in enumerate(ACTION_CONFIGS):
            print(f"\n[host] Action {action_idx+1}/4 — cpu={action['cpu_millicores']}mc mem={action['memory_limit_mb']}MB timeout={action['timeout_ms']}ms")

            state = build_state(instance["prompt"], code_features, instance["task_type"])

            # Run code only (no tests) first
            try:
                exec_result = send_code_to_guest(code_only, action["timeout_ms"])
            except Exception as e:
                print(f"[host] Execution error: {e}")
                exec_result = {"exit_code": -1, "timed_out": False, "wall_time_ms": 0,
                               "cpu_user_ms": 0, "cpu_sys_ms": 0, "mem_peak_kb": 0,
                               "stdout": "", "stderr": str(e)}

            print(f"[host]   exit={exec_result['exit_code']} wall={exec_result['wall_time_ms']}ms mem={exec_result['mem_peak_kb']}KB")

            # Run with tests if available
            tests_passed = None
            if has_tests and exec_result["exit_code"] == 0:
                try:
                    test_result = send_code_to_guest(code_with_tests, action["timeout_ms"])
                    tests_passed = (
                        test_result["exit_code"] == 0 and
                        "ALL_TESTS_PASSED" in test_result.get("stdout", "")
                    )
                    print(f"[host]   tests_passed={tests_passed} stderr={test_result.get('stderr','')[:100]}")
                except Exception as e:
                    print(f"[host]   test run error: {e}")
                    tests_passed = False

            reward_dict = compute_reward(exec_result, action, tests_passed)
            print(f"[host]   reward={reward_dict['r']} (success={reward_dict['r_success']} correctness={reward_dict['r_correctness']})")

            next_state = dict(state)
            success = 1 if exec_result["exit_code"] == 0 else 0
            next_state["recent_success_rate"]  = round(success * 0.1 + state["recent_success_rate"] * 0.9, 4)
            next_state["recent_mean_cpu_used"] = round(exec_result["cpu_user_ms"] * 0.1 + state["recent_mean_cpu_used"] * 0.9, 2)
            next_state["recent_mean_mem_used"] = round(exec_result["mem_peak_kb"] * 0.1 + state["recent_mean_mem_used"] * 0.9, 2)

            transition = {
                "episode_id": f"ep_{int(time.time())}_{inst_idx}_{action_idx}",
                "step": 0,
                "done": True,
                "dataset":   instance["dataset"],
                "task_id":   instance["task_id"],
                "state":     state,
                "action":    action,
                "execution": {
                    "exit_code":      exec_result["exit_code"],
                    "timed_out":      exec_result["timed_out"],
                    "wall_time_ms":   exec_result["wall_time_ms"],
                    "cpu_user_ms":    exec_result["cpu_user_ms"],
                    "cpu_sys_ms":     exec_result["cpu_sys_ms"],
                    "mem_peak_kb":    exec_result["mem_peak_kb"],
                    "stdout_snippet": exec_result.get("stdout", "")[:500],
                    "stderr_snippet": exec_result.get("stderr", "")[:500],
                    "tests_passed":   tests_passed,
                },
                "reward":            reward_dict["r"],
                "reward_components": {k: v for k, v in reward_dict.items() if k != "r"},
                "next_state":        next_state,
                "meta": {
                    "prompt_hash":       prompt_hash,
                    "code_hash":         code_hash,
                    "model":             OLLAMA_MODEL,
                    "prompt_tokens":     ollama_resp.get("prompt_eval_count", -1),
                    "completion_tokens": ollama_resp.get("eval_count", -1),
                    "llm_latency_ms":    llm_latency_ms,
                    "collected_at":      datetime.datetime.utcnow().isoformat() + "Z",
                    "collector_version": "v2.0-dataset",
                },
            }

            with open(OUTPUT_JSONL, "a") as f:
                f.write(json.dumps(transition) + "\n")
            transitions_written += 1

    stop_firecracker(fc_proc)
    print(f"\n[host] Done. {transitions_written} transitions written to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
