#!/usr/bin/env python3
"""
collect_codecontests.py — Collect RL transitions from CodeContests problems.

For each problem:
  1. Select test input: generated_tests (largest) → public_tests fallback.
  2. Boot PREP VM, run reference Python3 solution to verify expected output.
  3. Generate code via LLM: Laguna → Sonnet → Opus escalation.
  4. Run LLM code on PREP VM — if wrong output, refine (Laguna → Sonnet → Opus).
  5. Boot a fresh CONFIG VM per action config (14 configs).
  6. Execute LLM code, measure resources, compare output, save transitions.

Usage:
    python collect_codecontests.py --n 100
    python collect_codecontests.py --n 50 --min-rating 1600 --max-rating 1800
    python collect_codecontests.py --n 100 --skip-existing
"""

import argparse
import datetime
import json
import time
import uuid

from actions import ACTION_CONFIGS, static_analyse, build_state, compute_reward, sha8
from llm_cc import (
    build_cc_prompt, generate_cc_code, refine_cc_code,
    strip_code, SONNET_MODEL,
)

TRANS_PATH = "collected/codecontests_transitions.jsonl"
PAIRS_PATH = "collected/codecontests_code_pairs.jsonl"

MAX_REFINEMENT = 2


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _stress_level(cf_rating: int, difficulty: int) -> str:
    if cf_rating >= 1700 or difficulty >= 10:
        return "high"
    if cf_rating >= 1400 or difficulty >= 7:
        return "medium"
    return "low"


def _pick_test_input(row: dict) -> tuple[str, str]:
    """
    Returns (stdin, expected_output).
    Prefers generated_tests (largest input), falls back to public_tests.
    """
    gen_inputs  = row["generated_tests"]["input"]
    gen_outputs = row["generated_tests"]["output"]
    if gen_inputs:
        idx = max(range(len(gen_inputs)), key=lambda i: len(gen_inputs[i]))
        return gen_inputs[idx], gen_outputs[idx] if idx < len(gen_outputs) else ""

    pub_inputs  = row["public_tests"]["input"]
    pub_outputs = row["public_tests"]["output"]
    if pub_inputs:
        return pub_inputs[0], pub_outputs[0] if pub_outputs else ""

    return "", ""


def _get_py3_solutions(row: dict) -> list[str]:
    langs = row["solutions"]["language"]
    sols  = row["solutions"]["solution"]
    return [s for l, s in zip(langs, sols) if l == 3]


def stream_problems(min_rating: int, max_rating: int, n: int, skip: set) -> list:
    from datasets import load_dataset
    ds = load_dataset("deepmind/code_contests", split="train", streaming=True)

    problems = []
    for row in ds:
        if len(problems) >= n:
            break

        name = row["name"]
        if name in skip:
            continue

        cf_rating  = row["cf_rating"]
        difficulty = row["difficulty"]

        in_range = (
            (cf_rating >= min_rating and cf_rating <= max_rating)
            or (cf_rating == 0 and difficulty >= 9)
        )
        if not in_range:
            continue

        py3_sols = _get_py3_solutions(row)
        if not py3_sols:
            continue

        stdin, expected = _pick_test_input(row)
        if not stdin:
            continue

        problems.append({
            "name":        name,
            "description": row["description"],
            "cf_rating":   cf_rating,
            "difficulty":  difficulty,
            "cf_tags":     row["cf_tags"],
            "stdin":       stdin,
            "expected":    expected,
            "ref_solution": py3_sols[0],
            "stress_level": _stress_level(cf_rating, difficulty),
            "time_limit":  row.get("time_limit"),
            "memory_limit_bytes": row.get("memory_limit_bytes", 0),
        })
        print(f"[stream] queued {name} | cf_rating={cf_rating} | stress={problems[-1]['stress_level']} | input={len(stdin)} chars")

    return problems


# ── VM helpers ────────────────────────────────────────────────────────────────

def _boot_vm(vm_id: str, cpu_millicores: int, memory_mb: int):
    from vm import start_vm, wait_for_agent, apply_cgroups, stop_vm
    vm = start_vm(vm_id)
    time.sleep(3)
    if not wait_for_agent(vm):
        stop_vm(vm)
        return None
    apply_cgroups(vm, cpu_millicores, memory_mb)
    return vm


def _run_on_vm(vm, code: str, stdin: str, timeout_ms: int) -> dict:
    from vm import send_code
    wrapped = (
        f"import sys as _sys, io as _io\n"
        f"_sys.stdin = _io.StringIO({repr(stdin)})\n"
        + code
    )
    try:
        return send_code(vm, wrapped, timeout_ms)
    except Exception as e:
        return {
            "exit_code": -1, "timed_out": False, "wall_time_ms": 0,
            "cpu_user_ms": 0, "cpu_sys_ms": 0, "mem_peak_kb": 0,
            "stdout": "", "stderr": str(e),
        }


def _verify_ref_on_prep(ref_code: str, stdin: str, expected: str,
                         prep_config: dict) -> tuple[bool, str]:
    """
    Boot PREP VM, run reference solution, return (verified, actual_expected).
    If ref passes and expected was empty, returns ref's stdout as expected.
    """
    from vm import stop_vm
    vm_id = f"prep-ref-{uuid.uuid4().hex[:6]}"
    vm    = _boot_vm(vm_id, prep_config["cpu_millicores"], prep_config["memory_limit_mb"])
    if vm is None:
        print("[collect]   PREP VM failed to boot — trusting stored expected output")
        return True, expected

    result = _run_on_vm(vm, ref_code, stdin, prep_config["timeout_ms"])
    stop_vm(vm)

    if result["exit_code"] != 0:
        print(f"[collect]   ref solution failed on PREP VM (exit={result['exit_code']}) — trusting stored expected")
        return True, expected

    ref_out = result["stdout"].strip()
    if not expected:
        print(f"[collect]   expected was empty — using ref solution output ({len(ref_out)} chars)")
        return True, ref_out

    if ref_out == expected.strip():
        print(f"[collect]   ref verified ✓ ({len(ref_out)} chars)")
        return True, expected.strip()

    print(f"[collect]   ref output mismatch — using ref output as ground truth")
    return True, ref_out


def _check_llm_on_prep(code: str, stdin: str, expected: str,
                        prep_config: dict) -> tuple[dict, bool]:
    """
    Boot PREP VM, run LLM code, return (execution_result, passed).
    """
    from vm import stop_vm
    vm_id = f"prep-llm-{uuid.uuid4().hex[:6]}"
    vm    = _boot_vm(vm_id, prep_config["cpu_millicores"], prep_config["memory_limit_mb"])
    if vm is None:
        print("[collect]   PREP VM for LLM check failed to boot")
        return {"exit_code": -1, "timed_out": False, "wall_time_ms": 0,
                "cpu_user_ms": 0, "cpu_sys_ms": 0, "mem_peak_kb": 0,
                "stdout": "", "stderr": "boot failed"}, False

    result = _run_on_vm(vm, code, stdin, prep_config["timeout_ms"])
    stop_vm(vm)

    passed = (
        result["exit_code"] == 0 and
        result["stdout"].strip() == expected.strip()
    )
    return result, passed


# ── LLM generation + refinement ───────────────────────────────────────────────

def _generate_with_escalation(description: str, sample_input: str,
                               sample_output: str) -> tuple[str, str]:
    """
    Generate code: Laguna → Sonnet → Opus.
    Returns (code, model_used).
    """
    import requests, os

    prompt = build_cc_prompt(description, sample_input, sample_output)

    # Try Laguna first (free)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer":  "https://github.com/madiha/microforge-rl",
            },
            json={
                "model":       "poolside/laguna-m.1:free",
                "max_tokens":  8192,
                "temperature": 0.2,
                "messages":    [{"role": "user", "content": prompt}],
            },
            timeout=180,
        )
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            finish  = response.json()["choices"][0]["finish_reason"]
            if content and finish == "stop":
                code = strip_code(content)
                try:
                    import ast
                    ast.parse(code)
                    print(f"[llm_cc] Laguna succeeded ({len(code)} chars)")
                    return code, "laguna-m.1"
                except SyntaxError:
                    print("[llm_cc] Laguna syntax error — escalating to Sonnet")
            else:
                print(f"[llm_cc] Laguna returned None or hit length limit (finish={finish}) — escalating to Sonnet")
    except Exception as e:
        print(f"[llm_cc] Laguna error: {e} — escalating to Sonnet")

    # Escalate to Sonnet / Opus via generate_cc_code
    code, _, _ = generate_cc_code(prompt, retries=1)
    model_used = SONNET_MODEL
    return code, model_used


def _refine_with_escalation(code: str, stdin: str, expected: str,
                             actual: str, description: str,
                             prep_config: dict) -> str:
    """
    Refine wrong code on PREP VM. Laguna → Sonnet → Opus.
    Returns best code found (original if all fail).
    """
    import requests, os, ast as _ast
    from vm import stop_vm

    best_code = code

    for attempt in range(MAX_REFINEMENT):
        # Laguna refinement attempt
        if attempt == 0:
            prompt = (
                "The following Python code produced wrong output on a competitive programming problem.\n"
                f"Problem:\n{description[:800]}\n\n"
                f"Code:\n{code}\n\n"
                f"Input used:\n{stdin[:500]}\n\n"
                f"Expected output:\n{expected[:300]}\n\n"
                f"Actual output:\n{actual[:300]}\n\n"
                "Return only the corrected Python code. Read from stdin, write to stdout."
            )
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            try:
                resp = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Content-Type":  "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer":  "https://github.com/madiha/microforge-rl",
                    },
                    json={
                        "model": "poolside/laguna-m.1:free",
                        "max_tokens": 8192,
                        "temperature": 0.2,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=180,
                )
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    finish  = resp.json()["choices"][0]["finish_reason"]
                    if content and finish == "stop":
                        refined = strip_code(content)
                        try:
                            _ast.parse(refined)
                        except SyntaxError:
                            refined = ""
                    else:
                        refined = ""
                else:
                    refined = ""
            except Exception:
                refined = ""
        else:
            refined = refine_cc_code(code, stdin, expected, actual,
                                     description, attempt=attempt - 1)

        if not refined:
            continue

        # Check refined code on PREP VM
        exec_result, passed = _check_llm_on_prep(
            refined, stdin, expected, prep_config
        )
        actual = exec_result.get("stdout", "").strip()

        if passed:
            print(f"[collect]   refinement attempt {attempt + 1} passed ✓")
            return refined

        print(f"[collect]   refinement attempt {attempt + 1} still wrong")
        best_code = refined  # use latest refined even if wrong

    return best_code


# ── Per-problem collection ────────────────────────────────────────────────────

def process_cc_problem(problem: dict) -> dict | None:
    from vm import stop_vm

    highest = ACTION_CONFIGS[-1]
    prep_config = {
        "cpu_millicores":  highest["cpu_millicores"],
        "memory_limit_mb": highest["memory_limit_mb"],
        "timeout_ms":      30_000,
    }

    stdin    = problem["stdin"]
    expected = problem["expected"]
    desc     = problem["description"]
    name     = problem["name"]

    # ── Step 1: Verify reference solution on PREP VM ──────────────────────────
    print(f"[collect]   verifying reference solution on PREP VM...")
    _, expected = _verify_ref_on_prep(
        problem["ref_solution"], stdin, expected, prep_config
    )
    if not expected:
        print(f"[collect]   no expected output — skipping {name}")
        return None

    # ── Step 2: Generate code ─────────────────────────────────────────────────
    sample_in  = problem["stdin"][:400] if len(problem["stdin"]) > 400 else problem["stdin"]
    sample_out = expected[:200]
    print(f"[collect]   generating code...")
    generated_code, llm_model = _generate_with_escalation(desc, sample_in, sample_out)

    if not generated_code:
        print(f"[collect]   all LLM attempts failed — skipping {name}")
        return None

    # ── Step 3: Check LLM code on PREP VM ────────────────────────────────────
    print(f"[collect]   checking LLM code on PREP VM...")
    prep_exec, passed = _check_llm_on_prep(generated_code, stdin, expected, prep_config)
    actual_out = prep_exec.get("stdout", "").strip()

    if not passed:
        print(f"[collect]   LLM code wrong — refining...")
        generated_code = _refine_with_escalation(
            generated_code, stdin, expected, actual_out, desc, prep_config
        )

    # ── Step 4: Static analysis + state ──────────────────────────────────────
    code_features = static_analyse(generated_code)
    state         = build_state(desc, code_features, task_type=3)

    # ── Step 5: Run 14 config VMs ─────────────────────────────────────────────
    transitions = []
    for action_idx, action in enumerate(ACTION_CONFIGS):
        vm_id = f"cc-{uuid.uuid4().hex[:6]}"
        vm    = _boot_vm(vm_id, action["cpu_millicores"], action["memory_limit_mb"])

        if vm is None:
            print(f"[collect]   config {action_idx} VM failed to boot — skipping config")
            continue

        execution = _run_on_vm(vm, generated_code, stdin, action["timeout_ms"])
        stop_vm(vm)

        tests_passed = None
        if execution["exit_code"] == 0:
            tests_passed = (execution["stdout"].strip() == expected.strip())

        reward_dict = compute_reward(execution, action, tests_passed)

        success    = 1 if execution["exit_code"] == 0 else 0
        next_state = dict(state)
        next_state["recent_success_rate"]  = round(success * 0.1 + state["recent_success_rate"] * 0.9, 4)
        next_state["recent_mean_cpu_used"] = round(execution["cpu_user_ms"] * 0.1 + state["recent_mean_cpu_used"] * 0.9, 2)
        next_state["recent_mean_mem_used"] = round(execution["mem_peak_kb"] * 0.1 + state["recent_mean_mem_used"] * 0.9, 2)

        transitions.append({
            "episode_id": f"ep_{int(time.time())}_{name}_{action_idx}",
            "task_id":    f"cc_{name}",
            "dataset":    "codecontests",
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
                "llm_model":    llm_model,
                "stress_level": problem["stress_level"],
                "cf_rating":    problem["cf_rating"],
                "cf_tags":      problem["cf_tags"],
                "collected_at": datetime.datetime.utcnow().isoformat() + "Z",
            },
        })

        status = "✓" if tests_passed else ("TLE" if execution["timed_out"] else "✗")
        print(f"[collect]   cfg{action_idx:02d} cpu={action['cpu_millicores']}mc "
              f"mem={action['memory_limit_mb']}MB "
              f"exit={execution['exit_code']} wall={execution['wall_time_ms']}ms "
              f"mem={execution['mem_peak_kb']//1024}MB {status}")

    if not transitions:
        return None

    code_pair = {
        "task_id":          f"cc_{name}",
        "dataset":          "codecontests",
        "prompt":           desc,
        "generated_code":   generated_code,
        "reference_code":   problem["ref_solution"],
        "tests_passed":     transitions[-1]["execution"]["tests_passed"],
        "llm_model":        llm_model,
        "cf_rating":        problem["cf_rating"],
        "cf_tags":          problem["cf_tags"],
        "stress_level":     problem["stress_level"],
        "stdin_length":     len(stdin),
        "collected_at":     datetime.datetime.utcnow().isoformat() + "Z",
    }

    return {"transitions": transitions, "code_pair": code_pair}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int, default=100, help="Number of problems to collect")
    parser.add_argument("--min-rating",  type=int, default=1400)
    parser.add_argument("--max-rating",  type=int, default=1800)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    skip = set()
    if args.skip_existing:
        import os
        if os.path.exists(TRANS_PATH):
            with open(TRANS_PATH) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        # strip cc_ prefix to match problem name
                        skip.add(rec["task_id"].replace("cc_", ""))
                    except Exception:
                        pass
        print(f"[collect] --skip-existing: {len(skip)} already collected")

    print(f"[collect] Streaming CodeContests problems | rating={args.min_rating}-{args.max_rating} | n={args.n}")
    problems = stream_problems(args.min_rating, args.max_rating, args.n, skip)
    print(f"[collect] {len(problems)} problems queued | {len(problems) * len(ACTION_CONFIGS)} total VM boots")
    print(f"[collect] Transitions → {TRANS_PATH}")
    print(f"[collect] Code pairs  → {PAIRS_PATH}")

    transitions_written = 0
    for idx, problem in enumerate(problems):
        print(f"\n[collect] {'='*60}")
        print(f"[collect] {idx+1}/{len(problems)} | {problem['name']} | "
              f"cf_rating={problem['cf_rating']} | stress={problem['stress_level']} | "
              f"tags={problem['cf_tags']}")
        print(f"[collect] stdin={len(problem['stdin'])} chars | "
              f"expected={len(problem['expected'])} chars")

        try:
            result = process_cc_problem(problem)
        except Exception as e:
            print(f"[collect] ERROR: {e} — skipping")
            continue

        if result is None:
            continue

        with open(TRANS_PATH, "a") as f:
            for t in result["transitions"]:
                f.write(json.dumps(t) + "\n")
        with open(PAIRS_PATH, "a") as f:
            f.write(json.dumps(result["code_pair"]) + "\n")

        transitions_written += len(result["transitions"])
        passed = sum(1 for t in result["transitions"] if t["execution"]["tests_passed"])
        print(f"[collect] {idx+1} done | {passed}/{len(result['transitions'])} configs passed | "
              f"{transitions_written} transitions total")

    print(f"\n[collect] Done.")
    print(f"[collect]   {transitions_written} transitions → {TRANS_PATH}")
    print(f"[collect]   {len(problems)} code pairs  → {PAIRS_PATH}")


if __name__ == "__main__":
    main()
