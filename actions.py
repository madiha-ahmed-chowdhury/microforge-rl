"""
actions.py — RL domain logic: action configs, reward, state, dataset loading.

Pure Python — no subprocess, no network, no LLM calls.
"""

import ast
import datetime
import hashlib
import json
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

COLLECTED_DIR = WORK_DIR / "collected"
COLLECTED_DIR.mkdir(exist_ok=True)

# Legacy — used when no dataset is specified
OUTPUT_TRANSITIONS = str(COLLECTED_DIR / "rl_transitions_v2.jsonl")
OUTPUT_CODE_PAIRS  = str(COLLECTED_DIR / "rl_code_pairs_v2.jsonl")


def get_output_paths(dataset: str) -> tuple[str, str]:
    """Return (transitions_path, code_pairs_path) for a given dataset."""
    name = "effibench" if dataset in ("effibench", "effibench_large") else dataset
    return (
        str(COLLECTED_DIR / f"{name}_transitions.jsonl"),
        str(COLLECTED_DIR / f"{name}_code_pairs.jsonl"),
    )

# ── Action space ──────────────────────────────────────────────────────────────

ACTION_CONFIGS = [
    # label            cpu_mc  mem_mb  timeout_ms
    # 50mc variants — very very low CPU, isolating memory and timeout effects
    {"cpu_millicores":  50, "memory_limit_mb":  32, "timeout_ms":  1_000, "cpu_idx": 0, "mem_idx": 0, "timeout_idx": 0},
    {"cpu_millicores":  50, "memory_limit_mb":  48, "timeout_ms":  2_000, "cpu_idx": 0, "mem_idx": 1, "timeout_idx": 1},
    {"cpu_millicores":  50, "memory_limit_mb":  64, "timeout_ms":  3_000, "cpu_idx": 0, "mem_idx": 2, "timeout_idx": 2},
    # 80mc variants — very low CPU, different memory and timeout
    {"cpu_millicores":  80, "memory_limit_mb":  48, "timeout_ms":  1_500, "cpu_idx": 1, "mem_idx": 1, "timeout_idx": 3},
    {"cpu_millicores":  80, "memory_limit_mb":  64, "timeout_ms":  2_500, "cpu_idx": 1, "mem_idx": 2, "timeout_idx": 4},
    {"cpu_millicores":  80, "memory_limit_mb":  96, "timeout_ms":  4_000, "cpu_idx": 1, "mem_idx": 3, "timeout_idx": 5},
    # 100mc variants — low CPU, different memory and timeout
    {"cpu_millicores": 100, "memory_limit_mb":  64, "timeout_ms":  2_000, "cpu_idx": 2, "mem_idx": 2, "timeout_idx": 1},
    {"cpu_millicores": 100, "memory_limit_mb":  96, "timeout_ms":  3_500, "cpu_idx": 2, "mem_idx": 3, "timeout_idx": 6},
    {"cpu_millicores": 100, "memory_limit_mb": 128, "timeout_ms":  5_000, "cpu_idx": 2, "mem_idx": 4, "timeout_idx": 3},
    # 150mc variants
    {"cpu_millicores": 150, "memory_limit_mb":  64, "timeout_ms":  3_000, "cpu_idx": 3, "mem_idx": 2, "timeout_idx": 2},
    {"cpu_millicores": 150, "memory_limit_mb":  96, "timeout_ms":  5_000, "cpu_idx": 3, "mem_idx": 3, "timeout_idx": 3},
    # just okay
    {"cpu_millicores": 250, "memory_limit_mb": 128, "timeout_ms":  5_000, "cpu_idx": 4, "mem_idx": 4, "timeout_idx": 3},
    {"cpu_millicores": 250, "memory_limit_mb": 192, "timeout_ms":  8_000, "cpu_idx": 4, "mem_idx": 5, "timeout_idx": 7},
    # moderate — comfortable for most tasks
    {"cpu_millicores": 500, "memory_limit_mb": 256, "timeout_ms": 10_000, "cpu_idx": 5, "mem_idx": 6, "timeout_idx": 8},
]

# ── Dataset loaders ───────────────────────────────────────────────────────────

_JSON_DATASETS = {
    "security": WORK_DIR / "dataset_tasks" / "security_tasks.json",
}
_JSON_CACHE: dict[str, list] = {}

# ── EffiBench live loader (fetches directly from HuggingFace) ─────────────────

_HF_KW_HIGH = [
    "10^6", "10^5", "1000000", "100000", "graph", "tree", "dynamic programming",
    "dp", "matrix", "dijkstra", "floyd", "bellman", "shortest path",
    "minimum spanning", "topological", "strongly connected", "dfs", "bfs",
    "permutation", "backtrack", "segment tree", "fenwick", "binary indexed",
    "n^2", "n^3", "n²", "n³", "heap", "priority queue", "binary search",
    "knapsack", "subset sum", "combinatorics", "memoization", "1e6", "1e5",
]
_HF_KW_MED = [
    "sort", "search", "hash", "string", "recursion", "palindrome",
    "queue", "stack", "greedy", "two pointer", "sliding window", "priority",
]

_HF_CACHE: dict[str, list] = {}


def _hf_stress_level(desc: str) -> str:
    d = desc.lower()
    if any(k in d for k in _HF_KW_HIGH):
        return "high"
    if any(k in d for k in _HF_KW_MED):
        return "medium"
    return "low"


def _hf_get_python_solution(solutions: dict) -> str | None:
    if not solutions or not isinstance(solutions, dict):
        return None
    for key in ("python3", "python", "Python3", "Python"):
        sol = solutions.get(key)
        if sol and isinstance(sol, dict):
            return sol.get("code", "")
        if sol and isinstance(sol, str):
            return sol
    return None


def _hf_extract_test_cases(row: dict, max_cases: int = 3) -> list[dict]:
    raw = row.get("generated_tests") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []
    cases = []
    for tc in raw[:max_cases]:
        if isinstance(tc, dict):
            inp = str(tc.get("input", ""))
            out = str(tc.get("output", ""))
            if inp:
                cases.append({"input": inp, "output": out})
    return cases


def _hf_build_prompt(description: str, test_cases: list[dict]) -> str:
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


def _load_effibench_hf(large: bool = False) -> list:
    """Load all EffiBench problems with Python solutions directly from HuggingFace."""
    cache_key = "effibench_large" if large else "effibench"
    if cache_key in _HF_CACHE:
        return _HF_CACHE[cache_key]

    from datasets import load_dataset as _load_hf
    print("[actions] Loading EffiBench from HuggingFace cache...")
    ds = _load_hf("EffiBench/effibench-x", split="test")

    result = []
    for i, row in enumerate(ds):
        py_sol = _hf_get_python_solution(row.get("solutions") or {})
        if not py_sol:
            continue
        description = (row.get("description") or row.get("description_md") or "").strip()
        if not description:
            continue

        sl = _hf_stress_level(description)
        if large and sl == "low":
            continue  # effibench_large skips trivial problems

        test_cases = _hf_extract_test_cases(row)
        time_limit_ms   = (row.get("time_limit_nanos") or 0) // 1_000_000
        memory_limit_mb = (row.get("memory_limit_bytes") or 0) // (1024 * 1024)

        result.append({
            "dataset":             cache_key,
            "task_id":             f"effi_{i:04d}",
            "prompt":              _hf_build_prompt(description, test_cases),
            "code":                py_sol,
            "test_cases":          test_cases,
            "task_type":           3,
            "stress_level":        sl,
            "test_case_generator": row.get("test_case_generator") or "",
            "canonical_solution":  py_sol,
            "time_limit_ms":       time_limit_ms,
            "memory_limit_mb":     memory_limit_mb,
            "source_url":          row.get("url") or "",
            "tags":                row.get("tags") or [],
        })

    _HF_CACHE[cache_key] = result
    high = sum(1 for r in result if r["stress_level"] == "high")
    med  = sum(1 for r in result if r["stress_level"] == "medium")
    low  = sum(1 for r in result if r["stress_level"] == "low")
    has_gen = sum(1 for r in result if r["test_case_generator"])
    print(f"[actions] {len(result)} EffiBench problems | high={high} med={med} low={low} | {has_gen} have generators")
    return result


def _load_json_dataset(dataset: str) -> list:
    if dataset not in _JSON_CACHE:
        with open(_JSON_DATASETS[dataset]) as f:
            _JSON_CACHE[dataset] = json.load(f)
    return _JSON_CACHE[dataset]


def _json_row_to_instance(dataset: str, row: dict) -> dict:
    return {
        "dataset":           dataset,
        "task_id":           row.get("problem_id") or row.get("task_id", ""),
        "prompt":            row.get("prompt", ""),
        "code":              row.get("canonical_solution", ""),
        "test_cases":        row.get("test_cases", []),
        "task_type":         row.get("task_type", 3),
        "stress_input_path": row.get("stress_input_path", ""),
    }


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
    """Load multiple instances. n=None → all. offset → start index."""
    if dataset in ("effibench", "effibench_large"):
        rows = _load_effibench_hf(large=(dataset == "effibench_large"))
        end  = len(rows) if n is None else min(offset + n, len(rows))
        return rows[offset:end]
    if dataset in _JSON_DATASETS:
        rows = _load_json_dataset(dataset)
        end  = len(rows) if n is None else min(offset + n, len(rows))
        return [_json_row_to_instance(dataset, rows[i]) for i in range(offset, end)]
    table = _load_table(dataset)
    end   = table.num_rows if n is None else min(offset + n, table.num_rows)
    return [
        _row_to_instance(dataset, {col: table[col][i].as_py() for col in table.schema.names})
        for i in range(offset, end)
    ]


def load_one(dataset: str, idx: int) -> dict:
    """Load a single instance by index."""
    if dataset in ("effibench", "effibench_large"):
        rows = _load_effibench_hf(large=(dataset == "effibench_large"))
        if idx >= len(rows):
            raise IndexError(f"{dataset} only has {len(rows)} rows, got idx={idx}")
        return rows[idx]
    if dataset in _JSON_DATASETS:
        rows = _load_json_dataset(dataset)
        if idx >= len(rows):
            raise IndexError(f"{dataset} only has {len(rows)} rows, got idx={idx}")
        return _json_row_to_instance(dataset, rows[idx])
    table = _load_table(dataset)
    if idx >= table.num_rows:
        raise IndexError(f"{dataset} only has {table.num_rows} rows, got idx={idx}")
    row = {col: table[col][idx].as_py() for col in table.schema.names}
    return _row_to_instance(dataset, row)


def dataset_size(dataset: str) -> int:
    if dataset in ("effibench", "effibench_large"):
        return len(_load_effibench_hf(large=(dataset == "effibench_large")))
    if dataset in _JSON_DATASETS:
        return len(_load_json_dataset(dataset))
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
    r_resource    = -0.75 * cpu_waste - 0.75 * mem_waste
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
    code_with_tests adds test harness so correctness can be checked.

    MBPP/HumanEval: function-based — appends assert statements.
    EffiBench/security: stdin/stdout — prepends sys.stdin mock with test input,
                        then compares stdout to expected output.
    """
    dataset = instance["dataset"]

    if dataset == "mbpp":
        tests = "\n".join(instance.get("test_list", []))
        return code, code + "\n\n# --- tests ---\n" + tests + "\nprint('ALL_TESTS_PASSED')\n"

    elif dataset == "humaneval":
        entry = instance.get("entry_point", "")
        return code, (
            code + "\n\n"
            + instance.get("test_code", "") + "\n\n"
            + f"check({entry})\nprint('ALL_TESTS_PASSED')\n"
        )

    elif dataset in ("effibench", "effibench_large", "security"):
        test_cases = instance.get("test_cases", [])
        if not test_cases or not test_cases[0].get("input"):
            return code, code

        stdin_data = test_cases[0]["input"]

        expected_out = test_cases[0].get("output", "").strip()

        # code_only: mock stdin so the script doesn't block waiting for input
        stdin_mock = (
            "import sys as _sys, io as _io\n"
            f"_sys.stdin = _io.StringIO({repr(stdin_data)})\n"
        )
        code_only = stdin_mock + code

        # code_with_tests: capture stdout and compare to expected output
        test_harness = (
            "import sys as _sys, io as _io\n"
            f"_sys.stdin = _io.StringIO({repr(stdin_data)})\n"
            "_captured = _io.StringIO()\n"
            "_sys.stdout = _captured\n"
            + code + "\n"
            "_sys.stdout = _sys.__stdout__\n"
            f"_expected = {repr(expected_out)}\n"
            "_actual = _captured.getvalue().strip()\n"
            "if _actual == _expected:\n"
            "    print('ALL_TESTS_PASSED')\n"
            "else:\n"
            "    print(f'WRONG: expected={repr(_expected[:200])} got={repr(_actual[:200])}')\n"
        )
        return code_only, test_harness

    return code, code


# ── Core pipeline ─────────────────────────────────────────────────────────────

def process_instance(instance: dict, vm=None, fresh_vm_per_action: bool = False) -> dict:
    """
    Run one dataset instance through LLM + VM with all 14 action configs.

    Two modes:
      fresh_vm_per_action=False (default): pass a shared VMState via `vm`.
        All 14 executions run sequentially on the same VM (cgroups mutated between runs).
        Used by server.py for speed.
      fresh_vm_per_action=True: boots a brand-new VM for each of the 14 action configs,
        tears it down after, then boots the next. True isolation — each execution starts
        from a clean slate. Used by collecting_dataset.py for training data quality.

    Returns {"transitions": [...], "code_pair": {...}}.
    """
    import uuid
    from llm import call_minimax, build_llm_prompt, strip_code, MINIMAX_MODEL, CLAUDE_MODEL, HAIKU_MODEL
    from vm import send_code as vm_send_code

    llm_prompt, expected_func, prefill = build_llm_prompt(instance)
    llm_start  = time.monotonic()
    raw_text, prompt_tokens, completion_tokens = call_minimax(
        llm_prompt, expected_func=expected_func, prefill=prefill,
        stress_level=instance.get("stress_level", "low"),
    )
    llm_latency_ms = int((time.monotonic() - llm_start) * 1000)
    generated_code = strip_code(raw_text)
    sl = instance.get("stress_level", "low")
    actual_model = CLAUDE_MODEL if sl in ("high", "medium") else HAIKU_MODEL

    code_features           = static_analyse(generated_code)
    state                   = build_state(instance["prompt"], code_features, instance["task_type"])
    code_only, code_w_tests = build_runnable(generated_code, instance)
    has_tests = bool(
        instance.get("test_list") or
        instance.get("test_code") or
        (instance.get("test_cases") and instance["test_cases"][0].get("input"))
    )

    transitions = []
    for action_idx, action in enumerate(ACTION_CONFIGS):
        # Boot a fresh isolated VM for this action config if requested
        if fresh_vm_per_action:
            from vm import start_vm, stop_vm, wait_for_agent, apply_cgroups
            vm_id      = f"collect-{uuid.uuid4().hex[:6]}"
            current_vm = start_vm(vm_id)
            time.sleep(3)
            if not wait_for_agent(current_vm):
                stop_vm(current_vm)
                continue
            apply_cgroups(current_vm, action["cpu_millicores"], action["memory_limit_mb"])

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
            "episode_id": f"ep_{int(time.time())}_{instance['task_id']}_{action_idx}",
            "task_id":    instance["task_id"],
            "dataset":    instance["dataset"],
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
                "llm_model":    actual_model,
                "stress_level": sl,
                "collected_at": datetime.datetime.utcnow().isoformat() + "Z",
            },
        })

    code_pair = {
        "task_id":          instance["task_id"],
        "dataset":          instance["dataset"],
        "prompt":           instance["prompt"],
        "generated_code":   generated_code,
        "reference_code":   instance.get("code", ""),
        "tests_passed":     transitions[-1]["execution"]["tests_passed"],
        "llm_model":        actual_model,
        "prompt_tokens":    prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_latency_ms":   llm_latency_ms,
        "collected_at":     datetime.datetime.utcnow().isoformat() + "Z",
    }

    return {"transitions": transitions, "code_pair": code_pair}
