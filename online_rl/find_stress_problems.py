#!/usr/bin/env python3
"""
find_stress_problems.py

Streams CodeContests from HuggingFace, scores each problem by:
  1. memory_limit_bytes  (higher = more RAM allowed = heavier problem)
  2. max n from description constraints (bigger n = more work)
  3. largest generated test input size (actual input bulk)

Then runs the Python reference solution on the largest test input,
measures wall time + peak RSS, and saves the top 50 highest-stress
problems to online_rl/cc_pool_cache.json (appending to existing pool).

Usage:
    python3 online_rl/find_stress_problems.py [--top 50] [--scan 5000]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE_PATH = "online_rl/cc_pool_cache.json"


# ── Parse max-n from description ─────────────────────────────────────────────

def parse_max_n(desc: str) -> int:
    """Return the largest n constraint found in the description, or 0."""
    # Normalise non-breaking spaces
    text = desc.replace('\xa0', ' ').replace('·', '*').replace('×', '*')

    best = 0

    # Pattern: "n ≤ 2*10^5" or "n ≤ 2 * 10^5"
    for m in re.finditer(r'n\s*[≤<]=?\s*(\d+(?:\.\d+)?)\s*\*\s*10\^?\s*(\d+)', text, re.I):
        exp = min(int(m.group(2)), 18)
        val = float(m.group(1)) * 10 ** exp
        best = max(best, int(val))

    # Pattern: "n ≤ 10^6"
    for m in re.finditer(r'n\s*[≤<]=?\s*10\^?\s*(\d+)', text, re.I):
        exp = min(int(m.group(1)), 18)
        best = max(best, 10 ** exp)

    # Pattern: "n ≤ 200 000" or "n ≤ 200000"
    for m in re.finditer(r'n\s*[≤<]=?\s*([\d][\d\s]{0,8})', text, re.I):
        val_str = m.group(1).replace(' ', '')
        try:
            best = max(best, int(val_str))
        except ValueError:
            pass

    return best


# ── Run a script, measure wall time + peak RSS ────────────────────────────────

def run_and_measure(code: str, stdin: bytes, timeout: int = 20) -> dict:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        path = f.name
    try:
        t0 = time.monotonic()
        r = subprocess.run(
            ['/usr/bin/time', '-v', 'python3', path],
            input=stdin,
            capture_output=True,
            timeout=timeout,
        )
        wall_ms = int((time.monotonic() - t0) * 1000)
        mem_kb = 0
        for line in r.stderr.decode(errors='replace').splitlines():
            if 'Maximum resident' in line:
                mem_kb = int(line.split()[-1])
                break
        return {
            'wall_ms': wall_ms,
            'mem_kb': mem_kb,
            'ok': r.returncode == 0,
            'stdout': r.stdout.decode(errors='replace'),
        }
    except subprocess.TimeoutExpired:
        return {'wall_ms': timeout * 1000, 'mem_kb': 0, 'ok': False, 'stdout': ''}
    except Exception as e:
        return {'wall_ms': 0, 'mem_kb': 0, 'ok': False, 'stdout': ''}
    finally:
        os.unlink(path)


# ── Score a candidate ─────────────────────────────────────────────────────────



# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top',  type=int, default=50,
                        help='Number of stress problems to collect (default 50)')
    parser.add_argument('--scan', type=int, default=5000,
                        help='Max problems to scan from dataset (default 5000)')
    args = parser.parse_args()

    # Load existing cache so we skip duplicates
    existing_ids = set()
    cache = {'meta': {}, 'problems': []}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        existing_ids = {p['task_id'] for p in cache['problems']}
        print(f"[find] existing cache: {len(existing_ids)} problems")

    from datasets import load_dataset
    ds = load_dataset('deepmind/code_contests', split='train', streaming=True)

    candidates = []   # (score, problem_dict)
    scanned = 0

    print(f"[find] scanning up to {args.scan} problems from HuggingFace...")
    for row in ds:
        if scanned >= args.scan:
            break
        scanned += 1

        task_id = f"cc_{row['name']}"
        if task_id in existing_ids:
            continue

        # Must have a Python 3 solution
        py3 = [s for l, s in zip(row['solutions']['language'],
                                  row['solutions']['solution']) if l == 3]
        if not py3:
            continue

        # Must have generated or public tests
        gen_in  = row['generated_tests']['input']
        gen_out = row['generated_tests']['output']
        pub_in  = row['public_tests']['input']
        pub_out = row['public_tests']['output']

        if gen_in:
            idx      = max(range(len(gen_in)), key=lambda i: len(gen_in[i]))
            stdin    = gen_in[idx]
            expected = gen_out[idx] if idx < len(gen_out) else ''
        elif pub_in:
            stdin    = pub_in[0]
            expected = pub_out[0] if pub_out else ''
        else:
            continue

        tl_raw     = row.get('time_limit') or {}
        time_limit = tl_raw.get('seconds', 0) if isinstance(tl_raw, dict) else float(tl_raw or 0)
        if time_limit == 0:
            continue
        candidates.append((time_limit, {
            'task_id':      task_id,
            'description':  row['description'],
            'ref_solution': py3[0],
            'stdin':        stdin,
            'expected':     expected,
            'cf_rating':    row.get('cf_rating', 0),
            'cf_tags':      row.get('cf_tags', []),
            'time_limit':   time_limit,
        }))

        if scanned % 500 == 0:
            print(f"[find]   scanned {scanned} | candidates so far: {len(candidates)}")

    print(f"[find] scanned {scanned} problems, {len(candidates)} candidates with Python solutions")

    # Sort by time_limit descending, take top --top
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected_pairs = candidates[:args.top]
    selected = [p for _, p in selected_pairs]

    print(f"\n[find] selected {len(selected)} problems (sorted by time_limit)")
    print(f"\n{'#':>3}  {'Task':52} {'time_limit':>12} {'cf_rating':>10}")
    print('-' * 85)
    for i, p in enumerate(selected, 1):
        print(f"{i:>3}  {p['task_id'][:52]:52} {p['time_limit']:>12} {p['cf_rating']:>10}")

    # Append to cache
    for p in selected:
        # Remove internal measurement keys before saving
        clean = {k: v for k, v in p.items() if not k.startswith('_')}
        cache['problems'].append(clean)
        existing_ids.add(p['task_id'])

    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"\n[find] cache now has {len(cache['problems'])} problems ({len(selected)} new stress problems added)")


if __name__ == '__main__':
    main()
