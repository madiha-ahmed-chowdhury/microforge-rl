"""
actions.py — RL domain logic: action configs, reward, state, dataset loading.

Pure Python — no subprocess, no network, no LLM calls.
"""

import ast
import datetime
import hashlib
import math
import time
from pathlib import Path
from typing import Optional

import pyarrow.ipc as ipc

# ── Paths ─────────────────────────────────────────────────────────────────────

WORK_DIR = Path(__file__).parent.resolve()

MBPP_PATH = str(
    WORK_DIR / "datasets/google-research-datasets___mbpp/full/0.0.0"
    / "4bb6404fdc6cacfda99d4ac4205087b89d32030c/mbpp-test.arrow"
)
HUMANEVAL_PATH = str(
    WORK_DIR / "datasets/openai_humaneval/openai_humaneval/0.0.0"
    / "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval-test.arrow"
)

# Output files — written by collecting_dataset.py and server.py
OUTPUT_TRANSITIONS = str(WORK_DIR / "rl_transitions.jsonl")
OUTPUT_CODE_PAIRS  = str(WORK_DIR / "rl_code_pairs.jsonl")

# ── Action space ──────────────────────────────────────────────────────────────
# 4 resource allocation configs tried per dataset instance.
# Add / remove / tweak rows here to change the action space.

ACTION_CONFIGS = [
    {"cpu_millicores": 250,  "memory_limit_mb": 128, "timeout_ms":  5_000, "cpu_idx": 0, "mem_idx": 0, "timeout_idx": 0},
    {"cpu_millicores": 500,  "memory_limit_mb": 256, "timeout_ms": 10_000, "cpu_idx": 1, "mem_idx": 1, "timeout_idx": 1},
    {"cpu_millicores": 1000, "memory_limit_mb": 256, "timeout_ms": 10_000, "cpu_idx": 2, "mem_idx": 1, "timeout_idx": 1},
    {"cpu_millicores": 1000, "memory_limit_mb": 256, "timeout_ms": 30_000, "cpu_idx": 2, "mem_idx": 1, "timeout_idx": 2},
]

# ── Dataset loaders ───────────────────────────────────────────────────────────

def _load_table(dataset: str):
    path = MBPP_PATH if dataset == "mbpp" else HUMANEVAL_PATH
    with open(path, "rb") as f:
        return ipc.open_stream(f).read_all()


def _row_to_instance(dataset: str, row: dict) -> dict:
    if dataset == "mbpp":
        return {
            "dataset":   "mbpp",
            "task_id":   str(row["task_id"]),
            "prompt":    row["text"],
            "code":      row["code"],
            "test_list": row["test_list"],
            "task_type": 1,
        }
    return {
        "dataset":     "humaneval",
        "task_id":     row["task_id"],
        "prompt":      row["prompt"],
        "code":        row.get("canonical_solution", ""),
        "test_code":   row["test"],
        "entry_point": row["entry_point"],
        "task_type":   2,
    }


def load_all(dataset: str, n: Optional[int] = None, offset: int = 0) -> list:
    """
    Load multiple instances from the dataset.
    n=None → all rows. offset → start index.
    """
    table = _load_table(dataset)
    end   = table.num_rows if n is None else min(offset + n, table.num_rows)
    return [
        _row_to_instance(dataset, {col: table[col][i].as_py() for col in table.schema.names})
        for i in range(offset, end)
    ]


def load_one(dataset: str, idx: int) -> dict:
    """Load a single instance by index."""
    table = _load_table(dataset)
    if idx >= table.num_rows:
        raise IndexError(f"{dataset} only has {table.num_rows} rows, got idx={idx}")
    row = {col: table[col][idx].as_py() for col in table.schema.names}
    return _row_to_instance(dataset, row)


def dataset_size(dataset: str) -> int:
    return _load_table(dataset).num_rows


# ── RL helpers ────────────────────────────────────────────────────────────────

def sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def static_analyse(code: str) -> dict:
    """Extract static code features for the state vector."""
    features = {
        "line_count": 0, "cyclomatic_complexity": 1, "ast_node_count": 0,
        "has_recursion": 0, "has_external_calls": 0, "max_loop_depth": 0,
        "estimated_complexity": 3,
    }
    features["line_count"] = len(code.strip().splitlines())
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return features
    features["ast_node_count"] = sum(1 for _ in ast.walk(tree))
    branch_types = (
        ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With, ast.Assert,
        ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
    )
    features["cyclomatic_complexity"] = 1 + sum(
        1 for n in ast.walk(tree) if isinstance(n, branch_types)
    )
    def _max_depth(node, d=0):
        if isinstance(node, (ast.For, ast.While)):
            d += 1
        return max([d] + [_max_depth(c, d) for c in ast.iter_child_nodes(node)])
    features["max_loop_depth"] = _max_depth(tree)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    called  = {n.func.id for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    features["has_recursion"] = int(bool(defined & called))
    features["has_external_calls"] = int(
        any(k in code.lower() for k in ("subprocess", "requests", "urllib", "open(", "socket", "http"))
    )
    d = features["max_loop_depth"]
    features["estimated_complexity"] = 0 if d == 0 else 1 if d == 1 else 2 if d == 2 else 3
    return features


def build_state(prompt: str, code_features: dict, task_type: int) -> dict:
    """Build the RL state vector from prompt + code features."""
    words = prompt.split()
    return {
        "prompt_token_count":      len(prompt) // 4,
        "prompt_complexity_score": round(len(set(words)) / max(len(words), 1), 3),
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


def compute_reward(execution: dict, action: dict, tests_passed: Optional[bool]) -> dict:
    """Compute the RL reward from execution results."""
    r_success = 1.0 if execution["exit_code"] == 0 else -2.0
    r_timeout = -5.0 if execution["timed_out"] else 0.0
    wall_sec      = max(execution["wall_time_ms"] / 1000, 0.001)
    cpu_budget_ms = (action["cpu_millicores"] / 1000) * wall_sec * 1000
    cpu_used_ms   = execution["cpu_user_ms"] + execution["cpu_sys_ms"]
    cpu_waste     = max(0, (cpu_budget_ms - cpu_used_ms) / max(cpu_budget_ms, 1))
    mem_alloc_kb  = action["memory_limit_mb"] * 1024
    mem_waste     = max(0, (mem_alloc_kb - execution["mem_peak_kb"]) / max(mem_alloc_kb, 1))
    r_resource    = -0.3 * cpu_waste - 0.3 * mem_waste
    r_latency     = -0.2 * math.log(execution["wall_time_ms"] / 500 + 1)
    r_correctness = (1.0 if tests_passed is True else -1.0 if tests_passed is False else 0.0)
    total = r_success + r_timeout + r_resource + r_latency + r_correctness
    return {
        "r":             round(total, 4),
        "r_success":     round(r_success, 4),
        "r_timeout":     round(r_timeout, 4),
        "r_resource":    round(r_resource, 4),
        "r_latency":     round(r_latency, 4),
        "r_correctness": round(r_correctness, 4),
    }


def build_runnable(code: str, instance: dict) -> tuple[str, str]:
    """
    Returns (code_only, code_with_tests).
    code_with_tests appends the dataset's assertions so correctness can be checked.
    """
    if instance["dataset"] == "mbpp":
        tests = "\n".join(instance.get("test_list", []))
        return code, code + "\n\n# --- tests ---\n" + tests + "\nprint('ALL_TESTS_PASSED')\n"
    elif instance["dataset"] == "humaneval":
        entry = instance.get("entry_point", "")
        return code, (
            code + "\n\n"
            + instance.get("test_code", "") + "\n\n"
            + f"check({entry})\nprint('ALL_TESTS_PASSED')\n"
        )
    return code, code


# ── Core pipeline ─────────────────────────────────────────────────────────────

def process_instance(instance: dict, vm=None, fresh_vm_per_action: bool = False) -> dict:
    """
    Run one dataset instance through LLM + VM with all 4 action configs.

    Two modes:
      fresh_vm_per_action=False (default): pass a shared VMState via `vm`.
        All 4 executions run sequentially on the same VM (cgroups mutated between runs).
        Used by server.py for speed.
      fresh_vm_per_action=True: boots a brand-new VM for each of the 4 action configs,
        tears it down after, then boots the next. True isolation — each execution starts
        from a clean slate. Used by collecting_dataset.py for training data quality.

    Returns {"transitions": [...], "code_pair": {...}}.
    """
    import uuid
    from llm import call_minimax, build_llm_prompt, strip_code, MINIMAX_MODEL
    from vm import send_code as vm_send_code

    llm_prompt = build_llm_prompt(instance)
    llm_start  = time.monotonic()
    raw_text, prompt_tokens, completion_tokens = call_minimax(llm_prompt)
    llm_latency_ms = int((time.monotonic() - llm_start) * 1000)
    generated_code = strip_code(raw_text)

    code_features           = static_analyse(generated_code)
    state                   = build_state(instance["prompt"], code_features, instance["task_type"])
    code_only, code_w_tests = build_runnable(generated_code, instance)
    has_tests               = bool(instance.get("test_list") or instance.get("test_code"))

    transitions = []
    for action_idx, action in enumerate(ACTION_CONFIGS):
        # Boot a fresh isolated VM for this action config if requested
        if fresh_vm_per_action:
            from vm import start_vm, stop_vm, wait_for_agent
            vm_id      = f"collect-{uuid.uuid4().hex[:6]}"
            current_vm = start_vm(vm_id)
            time.sleep(3)
            if not wait_for_agent(current_vm):
                stop_vm(current_vm)
                print(f"[actions] WARNING: VM did not start for action_idx={action_idx}, skipping")
                continue
        else:
            current_vm = vm

        try:
            execution = vm_send_code(current_vm, code_only, action["timeout_ms"])
        except Exception as e:
            execution = {
                "exit_code": -1, "timed_out": False, "wall_time_ms": 0,
                "cpu_user_ms": 0, "cpu_sys_ms": 0, "mem_peak_kb": 0,
                "stdout": "", "stderr": str(e),
            }

        tests_passed = None
        if has_tests and execution["exit_code"] == 0:
            try:
                tr = vm_send_code(current_vm, code_w_tests, action["timeout_ms"])
                tests_passed = (
                    tr["exit_code"] == 0 and
                    "ALL_TESTS_PASSED" in tr.get("stdout", "")
                )
            except Exception:
                tests_passed = False

        if fresh_vm_per_action:
            stop_vm(current_vm)

        reward_dict = compute_reward(execution, action, tests_passed)
        next_state  = dict(state)
        success     = 1 if execution["exit_code"] == 0 else 0
        next_state["recent_success_rate"]  = round(success * 0.1 + state["recent_success_rate"] * 0.9, 4)
        next_state["recent_mean_cpu_used"] = round(execution["cpu_user_ms"] * 0.1 + state["recent_mean_cpu_used"] * 0.9, 2)
        next_state["recent_mean_mem_used"] = round(execution["mem_peak_kb"] * 0.1 + state["recent_mean_mem_used"] * 0.9, 2)

        transitions.append({
            "episode_id":     f"ep_{int(time.time())}_{instance['task_id']}_{action_idx}",
            "step":           0,
            "done":           True,
            "dataset":        instance["dataset"],
            "task_id":        instance["task_id"],
            "generated_code": generated_code,
            "state":          state,
            "action":         action,
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
                "collector_version": "v1.0",
            },
        })

    code_pair = {
        "task_id":        instance["task_id"],
        "dataset":        instance["dataset"],
        "prompt":         instance["prompt"],
        "generated_code": generated_code,
        "reference_code": instance.get("code", ""),
        "tests_passed":   transitions[-1]["execution"]["tests_passed"],
        "collected_at":   datetime.datetime.utcnow().isoformat() + "Z",
        "model":          MINIMAX_MODEL,
    }

    return {"transitions": transitions, "code_pair": code_pair}
