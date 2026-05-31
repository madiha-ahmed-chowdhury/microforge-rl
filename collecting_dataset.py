#!/usr/bin/env python3
"""
collecting_dataset.py — CLI script to collect RL transitions for a dataset.

For each dataset instance:
  1. [EffiBench only] Boots a prep VM, runs test_case_generator → fresh stress input + oracle output.
  2. Calls LLM to generate code (Claude for high-stress problems, Laguna otherwise).
  3. Boots a FRESH Firecracker microVM for each of the 14 resource configs.
  4. Executes generated code, measures metrics, tears VM down.
  5. Saves (state, action, reward, next_state) transitions and code pairs.

Usage:
    venv/bin/python collecting_dataset.py                         # all MBPP
    venv/bin/python collecting_dataset.py --dataset effibench     # all EffiBench (live from HF)
    venv/bin/python collecting_dataset.py --dataset effibench_large  # high/medium stress only
    venv/bin/python collecting_dataset.py --n 100                 # first 100
    venv/bin/python collecting_dataset.py --n 100 --offset 200    # rows 200-299
"""

import argparse
import json
import random
import time
import uuid

from actions import ACTION_CONFIGS, get_output_paths, load_all, process_instance
from llm import warmup_test


def _load_stratified(n: int, high_pct: float = 0.30, med_pct: float = 0.20) -> list:
    """Sample n problems from full EffiBench: 30% high, 20% medium, 50% low."""
    all_problems = load_all("effibench")

    by_level: dict[str, list] = {"high": [], "medium": [], "low": []}
    for p in all_problems:
        by_level[p["stress_level"]].append(p)

    n_high = int(n * high_pct)
    n_med  = int(n * med_pct)
    n_low  = n - n_high - n_med

    rng = random.Random(42)
    result = (
        rng.sample(by_level["high"],  min(n_high, len(by_level["high"])))
        + rng.sample(by_level["medium"], min(n_med,  len(by_level["medium"])))
        + rng.sample(by_level["low"],    min(n_low,  len(by_level["low"])))
    )
    rng.shuffle(result)

    high_got = sum(1 for p in result if p["stress_level"] == "high")
    med_got  = sum(1 for p in result if p["stress_level"] == "medium")
    low_got  = sum(1 for p in result if p["stress_level"] == "low")
    print(f"[collect] stratified sample: {len(result)} problems | high={high_got} med={med_got} low={low_got}")
    return result


def _generate_stress_input(instance: dict) -> None:
    """
    Run test_case_generator in a prep VM. If successful, overwrites instance["test_cases"]
    with [{"input": stdin_data, "output": canonical_output}].
    Falls back to pre-stored test_cases if anything fails.
    """
    generator_code = instance.get("test_case_generator", "")
    canonical_code = instance.get("canonical_solution") or instance.get("code", "")
    if not generator_code or not canonical_code:
        return

    from vm import apply_cgroups, send_code, start_vm, stop_vm, wait_for_agent

    highest = ACTION_CONFIGS[-1]
    vm_id   = f"prep-{uuid.uuid4().hex[:6]}"
    vm      = start_vm(vm_id)
    time.sleep(3)
    if not wait_for_agent(vm):
        stop_vm(vm)
        print("[collect]   prep VM failed — using pre-stored test cases")
        return
    apply_cgroups(vm, highest["cpu_millicores"], highest["memory_limit_mb"])

    # Generate multiple cases across multiple seeds, pick the largest input.
    # Monkey-patch random.randint to bias toward large values so generators
    # that do n = random.randint(1, 100000) produce maximum-size inputs.
    gen_script = f"""
import sys as _sys
import random as _random

_orig_randint = _random.randint
def _max_randint(a, b):
    return b if b >= 100 else _orig_randint(a, b)
_random.randint = _max_randint

_orig_randrange = _random.randrange
def _max_randrange(start, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0
    return (stop - step) if (stop - start) >= 100 else _orig_randrange(start, stop, step)
_random.randrange = _max_randrange

{generator_code}
best = ''
for _seed in [42, 123, 999, 7777, 31337]:
    try:
        cases = generate_test_cases(num_cases=3, seed=_seed)
        for tc in (cases or []):
            inp = str(tc.get('input', '') if isinstance(tc, dict) else '')
            if len(inp) > len(best):
                best = inp
    except Exception as e:
        _sys.stderr.write(f'GEN_ERR seed={{_seed}}: {{e}}\\n')
print(best, end='')
"""
    try:
        gen_result = send_code(vm, gen_script, 45_000)
    except Exception as e:
        stop_vm(vm)
        print(f"[collect]   prep VM crashed during generator: {e} — using pre-stored test cases")
        return
    stdin_data = gen_result.get("stdout", "").strip()

    if not stdin_data:
        stop_vm(vm)
        print("[collect]   generator produced no output — using pre-stored test cases")
        return

    stdin_mock = f"import sys as _sys, io as _io\n_sys.stdin = _io.StringIO({repr(stdin_data)})\n"
    ref_result = send_code(vm, stdin_mock + canonical_code, 15_000)
    stop_vm(vm)

    if ref_result["exit_code"] == 0:
        ref_output = ref_result.get("stdout", "").strip()
        instance["test_cases"] = [{"input": stdin_data, "output": ref_output}]
        print(f"[collect]   stress input: {len(stdin_data)} chars | expected output: {len(ref_output)} chars")
    else:
        print(f"[collect]   canonical failed (exit={ref_result['exit_code']}) — using pre-stored test cases")


def main():
    warmup_test()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        choices=["mbpp", "humaneval", "effibench", "effibench_large", "security"],
                        default="mbpp")
    parser.add_argument("--n",          type=int, default=None, help="Number of instances (default: all)")
    parser.add_argument("--offset",     type=int, default=0,    help="Start from this dataset index")
    parser.add_argument("--test",       action="store_true",    help="Run 1 high + 1 medium + 1 low stress problem as a sanity check")
    parser.add_argument("--stratified", action="store_true",    help="Sample 30%% high / 20%% medium / 50%% low from full EffiBench (requires --n)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip task_ids already present in the output file")
    args = parser.parse_args()

    if args.test:
        all_problems = load_all("effibench")
        instances = []
        for level in ("high", "medium", "low"):
            match = next((p for p in all_problems if p["stress_level"] == level), None)
            if match:
                instances.append(match)
                print(f"[collect] test: picked {level} stress → task_id={match['task_id']}")
    elif args.stratified:
        if not args.n:
            print("[collect] ERROR: --stratified requires --n")
            return
        instances = _load_stratified(args.n)
        instances = instances[args.offset:]
    else:
        instances = load_all(args.dataset, args.n, args.offset)
    trans_path, pairs_path = get_output_paths("effibench" if (args.test or args.stratified) else args.dataset)

    if args.skip_existing:
        import os
        already = set()
        if os.path.exists(trans_path):
            import json as _json
            with open(trans_path) as _f:
                for _line in _f:
                    try: already.add(_json.loads(_line)["task_id"])
                    except: pass
        before = len(instances)
        instances = [i for i in instances if i["task_id"] not in already]
        print(f"[collect] --skip-existing: skipped {before - len(instances)} already-collected, {len(instances)} remaining")

    n_configs = len(ACTION_CONFIGS)
    print(f"[collect] {len(instances)} instances from {args.dataset} (offset={args.offset})")
    print(f"[collect] Fresh VM per action config — {len(instances) * n_configs} total VM boots")
    print(f"[collect] Transitions → {trans_path}")
    print(f"[collect] Code pairs  → {pairs_path}")

    transitions_written = 0
    for idx, instance in enumerate(instances):
        print(f"\n[collect] {'='*55}")
        print(f"[collect] {idx+1}/{len(instances)} | {instance['dataset']} | task_id={instance['task_id']} | stress={instance.get('stress_level', 'n/a')}")
        print(f"[collect] Prompt: {instance['prompt'][:100]}...")

        if instance.get("test_case_generator"):
            print(f"[collect] Running generator in prep VM...")
            try:
                _generate_stress_input(instance)
            except Exception as e:
                print(f"[collect]   generator step failed: {e} — using pre-stored test cases")

        if not instance.get("test_cases") or not instance["test_cases"][0].get("input"):
            print(f"[collect] Skipping — no usable test input (generator failed + no pre-stored cases)")
            continue

        try:
            result = process_instance(instance, fresh_vm_per_action=True)
        except Exception as e:
            print(f"[collect] ERROR: {e} — skipping")
            continue

        with open(trans_path, "a") as f:
            for t in result["transitions"]:
                f.write(json.dumps(t) + "\n")
        with open(pairs_path, "a") as f:
            f.write(json.dumps(result["code_pair"]) + "\n")

        transitions_written += len(result["transitions"])
        print(f"[collect] Written {transitions_written} transitions so far")

    print(f"\n[collect] Done.")
    print(f"[collect]   {transitions_written} transitions → {trans_path}")
    print(f"[collect]   {len(instances)} code pairs  → {pairs_path}")


if __name__ == "__main__":
    main()
