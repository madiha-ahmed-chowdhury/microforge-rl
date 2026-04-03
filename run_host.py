#!/usr/bin/env python3
"""
run_host.py — Host orchestrator for one offline RL data collection episode.
Modified to use Ollama (local, free, CPU-compatible) instead of Anthropic API.

Install Ollama first:
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull qwen2.5-coder:3b

Then run:
  python3 run_host.py
"""

import os
import sys
import json
import math
import time
import socket
import hashlib
import subprocess
import tempfile
import signal
import datetime
import ast
import re
import urllib.request
import urllib.error
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

WORK_DIR        = Path(__file__).parent.resolve()

FIRECRACKER_BIN = str(WORK_DIR / "firecracker")
VM_CONFIG       = str(WORK_DIR / "vm_config.json")
VSOCK_SOCK      = str(WORK_DIR / "vsock.sock")   # must match vm_config.json
OUTPUT_JSONL    = str(WORK_DIR / "transitions.jsonl")

# Ollama config — no API key needed, runs fully locally
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "qwen2.5-coder:3b"   # good CPU model; ~2GB RAM
# Other good CPU options (pull with: ollama pull <name>):
#   "phi3.5:mini"      — Microsoft, ~2.2GB, very fast on CPU
#   "llama3.2:3b"      — Meta, ~2GB, general purpose
#   "deepseek-coder:1.3b" — tiny, fastest on weak CPUs, code-focused

# vsock port the guest agent listens on (must match agent.py)
VSOCK_PORT      = 52

AGENT_BOOT_TIMEOUT = 180   # seconds to wait for guest agent after VM boots

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE DATASET INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_INSTANCE = {
    "prompt": (
        "Write a Python function called 'find_duplicates' that takes a list "
        "of integers and returns a sorted list of any integers that appear "
        "more than once. Include a small test at the bottom that prints the result "
        "of calling find_duplicates([1,2,3,2,4,3,5]). "
        "Return only the code, no explanation."
    ),
    "task_type": 3,
    "action": {
        "cpu_millicores": 500,
        "memory_limit_mb": 256,
        "timeout_ms": 10000,
        "prompt_variant": 0,
        "cpu_idx": 1,
        "mem_idx": 1,
        "timeout_idx": 2,
        "variant_idx": 0,
    },
    "latency_baseline_ms": 500,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def count_prompt_tokens_approx(text: str) -> int:
    return len(text) // 4


def call_ollama(prompt: str, timeout_sec: int = 120) -> dict:
    """
    Call the local Ollama server and return the parsed JSON response.
    Raises urllib.error.URLError if Ollama is not running.
    """
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,   # wait for full response, simpler to parse
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        print(
            "\n[host] ERROR: Could not reach Ollama at http://localhost:11434\n"
            "       Make sure Ollama is running:  ollama serve\n"
            "       And the model is pulled:      ollama pull " + OLLAMA_MODEL + "\n"
        )
        raise


def static_analyse(code: str) -> dict:
    features = {
        "line_count": 0,
        "cyclomatic_complexity": 1,
        "ast_node_count": 0,
        "has_recursion": 0,
        "has_external_calls": 0,
        "max_loop_depth": 0,
        "estimated_complexity": 3,
    }

    lines = code.strip().splitlines()
    features["line_count"] = len(lines)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return features

    features["ast_node_count"] = sum(1 for _ in ast.walk(tree))

    branch_types = (
        ast.If, ast.For, ast.While, ast.ExceptHandler,
        ast.With, ast.Assert,
        ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
    )
    branches = sum(1 for node in ast.walk(tree) if isinstance(node, branch_types))
    features["cyclomatic_complexity"] = 1 + branches

    def max_loop_depth(node, depth=0):
        if isinstance(node, (ast.For, ast.While)):
            depth += 1
        return max(
            [depth] + [max_loop_depth(child, depth) for child in ast.iter_child_nodes(node)]
        )
    features["max_loop_depth"] = max_loop_depth(tree)

    defined_funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    called_names  = {n.func.id for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    features["has_recursion"] = int(bool(defined_funcs & called_names))

    external_indicators = {"subprocess", "requests", "urllib", "open", "socket", "http"}
    source_lower = code.lower()
    features["has_external_calls"] = int(
        any(ind in source_lower for ind in external_indicators)
    )

    if features["max_loop_depth"] == 0:
        features["estimated_complexity"] = 0
    elif features["max_loop_depth"] == 1:
        features["estimated_complexity"] = 1
    elif features["max_loop_depth"] == 2:
        features["estimated_complexity"] = 2
    else:
        features["estimated_complexity"] = 3

    return features


def build_state(prompt: str, code_features: dict, instance: dict) -> dict:
    return {
        "prompt_token_count":       count_prompt_tokens_approx(prompt),
        "prompt_complexity_score":  round(len(set(prompt.split())) / max(len(prompt.split()), 1), 3),
        "task_type":                instance["task_type"],
        "has_loops_hint":           int("loop" in prompt.lower() or "repeat" in prompt.lower()),
        "has_io_hint":              int("file" in prompt.lower() or "read" in prompt.lower() or "write" in prompt.lower()),
        "example_count":            0,
        **code_features,
        "host_cpu_load_1m":         0.0,
        "host_mem_available_mb":    512,
        "queue_depth":              0,
        "recent_success_rate":      1.0,
        "recent_mean_cpu_used":     0.0,
        "recent_mean_mem_used":     0.0,
    }


def compute_reward(execution: dict, action: dict, latency_baseline_ms: int) -> dict:
    r_success = 1.0 if execution["exit_code"] == 0 else -2.0
    r_timeout = -5.0 if execution["timed_out"] else 0.0

    wall_sec       = max(execution["wall_time_ms"] / 1000, 0.001)
    cpu_budget_ms  = (action["cpu_millicores"] / 1000) * wall_sec * 1000
    cpu_used_ms    = execution["cpu_user_ms"] + execution["cpu_sys_ms"]
    cpu_waste      = max(0, (cpu_budget_ms - cpu_used_ms) / max(cpu_budget_ms, 1))

    mem_alloc_kb   = action["memory_limit_mb"] * 1024
    mem_used_kb    = execution["mem_peak_kb"]
    mem_waste      = max(0, (mem_alloc_kb - mem_used_kb) / max(mem_alloc_kb, 1))

    r_resource = -0.3 * cpu_waste - 0.3 * mem_waste
    r_latency  = -0.2 * math.log(execution["wall_time_ms"] / latency_baseline_ms + 1)

    total = r_success + r_timeout + r_resource + r_latency

    return {
        "r":          round(total, 4),
        "r_success":  round(r_success, 4),
        "r_timeout":  round(r_timeout, 4),
        "r_resource": round(r_resource, 4),
        "r_latency":  round(r_latency, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIRECRACKER MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
def start_firecracker() -> subprocess.Popen:
    if os.path.exists(VSOCK_SOCK):
        os.remove(VSOCK_SOCK)
    api_socket = str(WORK_DIR / "firecracker.socket")
    if os.path.exists(api_socket):
        os.remove(api_socket)

    cmd = [
        FIRECRACKER_BIN,
        "--api-sock", api_socket,
        "--config-file", VM_CONFIG,
        "--log-path", str(WORK_DIR / "firecracker.log"),
        "--level", "Info",
    ]
    print("[host] Starting Firecracker...")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

def _vsock_connect(timeout_sec: float) -> socket.socket:
    """
    Connect to the Firecracker vsock UDS proxy.
    Firecracker exposes guest vsock via a Unix domain socket, not AF_VSOCK.
    Protocol: connect to UDS, send "CONNECT {port}\\n", read "OK {host_port}\\n".
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_sec)
    sock.connect(VSOCK_SOCK)
    sock.sendall(f"CONNECT {VSOCK_PORT}\n".encode())
    # Read the "OK <host_port>\n" handshake response
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
    print(f"[host] Waiting up to {timeout_sec}s for guest agent to start...")
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
            data = json.loads(response.decode())
            if data.get("exit_code") == 0:
                print("[host] Guest agent is ready.")
                return True
        except (ConnectionRefusedError, OSError, json.JSONDecodeError, socket.timeout):
            time.sleep(1)

    return False


def send_code_to_guest(code: str, timeout_ms: int) -> dict:
    print(f"[host] Sending {len(code)} chars of code to guest (timeout={timeout_ms}ms)...")
    sock = _vsock_connect(timeout_ms / 1000 + 15)
    sock.settimeout(timeout_ms / 1000 + 15)

    request = json.dumps({"code": code, "timeout": timeout_ms // 1000}) + "\n"
    sock.sendall(request.encode("utf-8"))

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
    instance = SAMPLE_INSTANCE
    action   = instance["action"]

    # ── Phase 1: Build pre-code state ──────────────────────────────────────
    print("\n[host] === Phase 1: Building pre-execution state ===")
    pre_code_features = static_analyse("")
    state_partial = build_state(instance["prompt"], pre_code_features, instance)
    print(f"[host] Prompt token count: {state_partial['prompt_token_count']}")

    # ── Phase 2: Call Ollama to generate code ──────────────────────────────
    print(f"\n[host] === Phase 2: Calling Ollama ({OLLAMA_MODEL}) for code generation ===")
    llm_start    = time.monotonic()
    ollama_resp  = call_ollama(instance["prompt"])
    llm_latency_ms  = int((time.monotonic() - llm_start) * 1000)
    generated_code  = ollama_resp["response"]
    # Strip markdown fences and any prose after the code
    # Strip markdown fences and prose from Ollama output
    import re
    fenced = re.search(r'```(?:python)?\n?(.*?)```', generated_code, re.DOTALL)
    if fenced:
        generated_code = fenced.group(1).strip()
    else:
        # No fences — drop lines that are clearly prose sentences
        clean = []
        for line in generated_code.splitlines():
            s = line.strip()
            # A line is prose if it starts with a capital letter, has no code chars, and isn't a comment
            is_prose = (s and s[0].isupper() and '=' not in s and '(' not in s and ':' not in s and not s.startswith('#'))
            if not is_prose:
                clean.append(line)
        generated_code = '\n'.join(clean).strip()
    prompt_hash     = sha8(instance["prompt"])
    code_hash       = sha8(generated_code)

    print(f"[host] Ollama responded in {llm_latency_ms}ms")
    print(f"[host] Generated code ({len(generated_code)} chars):")
    print("─" * 60)
    print(generated_code[:500], "..." if len(generated_code) > 500 else "")
    print("─" * 60)

    # ── Phase 3: Static analysis ───────────────────────────────────────────
    print("\n[host] === Phase 3: Static analysis ===")
    code_features = static_analyse(generated_code)
    print(f"[host] Code features: {json.dumps(code_features, indent=2)}")
    state = build_state(instance["prompt"], code_features, instance)

    # ── Phase 4: Start Firecracker ─────────────────────────────────────────
    print("\n[host] === Phase 4: Starting microVM ===")
    fc_proc = start_firecracker()
    time.sleep(3)   # give VM time to initialize before polling
    agent_ready = wait_for_agent()

    if not agent_ready:
        print("[host] ERROR: Guest agent did not start in time.")
        stop_firecracker(fc_proc)
        sys.exit(1)

    # ── Phase 5: Execute code in microVM ───────────────────────────────────
    print("\n[host] === Phase 5: Executing code in microVM ===")
    try:
        execution = send_code_to_guest(generated_code, action["timeout_ms"])
    except Exception as e:
        print(f"[host] ERROR executing in guest: {e}")
        execution = {
            "exit_code": -1, "timed_out": False, "wall_time_ms": 0,
            "cpu_user_ms": 0, "cpu_sys_ms": 0, "mem_peak_kb": 0,
            "stdout": "", "stderr": str(e),
        }

    print(f"[host] Execution result:")
    print(f"  exit_code    = {execution['exit_code']}")
    print(f"  timed_out    = {execution['timed_out']}")
    print(f"  wall_time_ms = {execution['wall_time_ms']}")
    print(f"  cpu_user_ms  = {execution['cpu_user_ms']}")
    print(f"  cpu_sys_ms   = {execution['cpu_sys_ms']}")
    print(f"  mem_peak_kb  = {execution['mem_peak_kb']}")
    print(f"  stdout       = {execution.get('stdout','')[:200]}")
    print(f"  stderr       = {execution.get('stderr','')[:200]}")

    # ── Phase 6: Compute reward ────────────────────────────────────────────
    print("\n[host] === Phase 6: Computing reward ===")
    reward_dict = compute_reward(execution, action, instance["latency_baseline_ms"])
    print(f"[host] Reward: {json.dumps(reward_dict, indent=2)}")

    # ── Phase 7: Build next_state ──────────────────────────────────────────
    next_state = dict(state)
    success = 1 if execution["exit_code"] == 0 else 0
    next_state["recent_success_rate"]  = round(success * 0.1 + state["recent_success_rate"] * 0.9, 4)
    next_state["recent_mean_cpu_used"] = round(execution["cpu_user_ms"] * 0.1 + state["recent_mean_cpu_used"] * 0.9, 2)
    next_state["recent_mean_mem_used"] = round(execution["mem_peak_kb"] * 0.1 + state["recent_mean_mem_used"] * 0.9, 2)

    # ── Phase 8: Write transition record ───────────────────────────────────
    print("\n[host] === Phase 7: Writing transition record ===")

    transition = {
        "episode_id": f"ep_{int(time.time())}",
        "step": 0,
        "done": True,

        "state": state,
        "action": action,

        "execution": {
            "exit_code":      execution["exit_code"],
            "timed_out":      execution["timed_out"],
            "wall_time_ms":   execution["wall_time_ms"],
            "cpu_user_ms":    execution["cpu_user_ms"],
            "cpu_sys_ms":     execution["cpu_sys_ms"],
            "mem_peak_kb":    execution["mem_peak_kb"],
            "stdout_snippet": execution.get("stdout", "")[:500],
            "stderr_snippet": execution.get("stderr", "")[:500],
        },

        "reward":            reward_dict["r"],
        "reward_components": {k: v for k, v in reward_dict.items() if k != "r"},

        "next_state": next_state,

        "meta": {
            "prompt_hash":       prompt_hash,
            "code_hash":         code_hash,
            "model":             OLLAMA_MODEL,
            # Ollama returns token counts in eval_count / prompt_eval_count
            "prompt_tokens":     ollama_resp.get("prompt_eval_count", -1),
            "completion_tokens": ollama_resp.get("eval_count", -1),
            "llm_latency_ms":    llm_latency_ms,
            "collected_at":      datetime.datetime.utcnow().isoformat() + "Z",
            "collector_version": "v1.1-ollama",
        },
    }

    with open(OUTPUT_JSONL, "a") as f:
        f.write(json.dumps(transition) + "\n")

    print(f"[host] Transition written to {OUTPUT_JSONL}")
    print(f"\n[host] Full transition record:")
    print(json.dumps(transition, indent=2))

    # ── Phase 9: Shutdown ──────────────────────────────────────────────────
    stop_firecracker(fc_proc)
    print("\n[host] Done. One transition collected.")


if __name__ == "__main__":
    main()