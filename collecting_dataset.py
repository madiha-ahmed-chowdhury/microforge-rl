#!/usr/bin/env python3
"""
collecting_dataset.py — CLI script to collect RL transitions for a dataset.

For each dataset instance:
  1. Calls MiniMax LLM to generate Python code from the prompt.
  2. Boots a FRESH Firecracker microVM for each of the 4 resource allocation configs.
  3. Executes the generated code inside that VM, measures metrics.
  4. Tears the VM down — clean slate for the next execution.
  5. Saves (state, action, reward, next_state) transitions and code pairs.

Using a fresh VM per execution ensures true isolation: each (code, resource_config)
pair is measured independently with no leftover filesystem or memory state.

Output:
    rl_transitions.jsonl  — (s, a, r, s', done) transitions, 4 per instance
    rl_code_pairs.jsonl   — prompt + generated code + reference code

Usage:
    venv/bin/python collecting_dataset.py                         # all MBPP
    venv/bin/python collecting_dataset.py --dataset humaneval     # all HumanEval
    venv/bin/python collecting_dataset.py --n 100                 # first 100
    venv/bin/python collecting_dataset.py --n 100 --offset 200    # rows 200-299
"""

import argparse
import json
import sys

from actions import OUTPUT_CODE_PAIRS, OUTPUT_TRANSITIONS, load_all, process_instance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mbpp", "humaneval"], default="mbpp")
    parser.add_argument("--n",      type=int, default=None, help="Number of instances (default: all)")
    parser.add_argument("--offset", type=int, default=0,    help="Start from this dataset index")
    args = parser.parse_args()

    instances = load_all(args.dataset, args.n, args.offset)
    print(f"[collect] {len(instances)} instances from {args.dataset} (offset={args.offset})")
    print(f"[collect] Fresh VM per action config — {len(instances) * 4} total VM boots")
    print(f"[collect] Transitions → {OUTPUT_TRANSITIONS}")
    print(f"[collect] Code pairs  → {OUTPUT_CODE_PAIRS}")

    transitions_written = 0
    for idx, instance in enumerate(instances):
        print(f"\n[collect] {'='*55}")
        print(f"[collect] {idx+1}/{len(instances)} | {instance['dataset']} | task_id={instance['task_id']}")
        print(f"[collect] Prompt: {instance['prompt'][:100]}...")

        try:
            result = process_instance(instance, fresh_vm_per_action=True)
        except Exception as e:
            print(f"[collect] ERROR: {e} — skipping")
            continue

        with open(OUTPUT_TRANSITIONS, "a") as f:
            for t in result["transitions"]:
                f.write(json.dumps(t) + "\n")
        with open(OUTPUT_CODE_PAIRS, "a") as f:
            f.write(json.dumps(result["code_pair"]) + "\n")

        transitions_written += len(result["transitions"])
        print(f"[collect] Written {transitions_written} transitions so far")

    print(f"\n[collect] Done.")
    print(f"[collect]   {transitions_written} transitions → {OUTPUT_TRANSITIONS}")
    print(f"[collect]   {len(instances)} code pairs  → {OUTPUT_CODE_PAIRS}")


if __name__ == "__main__":
    main()
