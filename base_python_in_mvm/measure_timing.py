import sys
import json
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from online_rl.vm_runner import boot_vm, run_code_on_vm
from vm import stop_vm

MEMORY_MB   = 128
CPU_MC      = 100
TIMEOUT_MS  = 10000
RUNS        = 5

PROGRAMS = [
    {
        "name":  "level0_trivial",
        "stdin": "",
        "code":  "a = 1 + 1\nprint(a)\n",
    },
    {
        "name":  "level1_imports_only",
        "stdin": "",
        "code": (
            "import sys, math, collections, itertools, functools\n"
            "input = sys.stdin.readline\n"
            "print('ok')\n"
        ),
    },
    {
        "name":  "level2_read_and_sum",
        "stdin": "5\n1\n2\n3\n4\n5\n",
        "code": (
            "import sys\n"
            "input = sys.stdin.readline\n"
            "n = int(input())\n"
            "print(sum(int(input()) for _ in range(n)))\n"
        ),
    },
    {
        "name":  "level3_typical_easy_cp",
        "stdin": "4\n3 1 4 1\n",
        "code": (
            "import sys\n"
            "from collections import defaultdict, Counter\n"
            "import math\n"
            "input = sys.stdin.readline\n"
            "\n"
            "n = int(input())\n"
            "a = list(map(int, input().split()))\n"
            "freq = Counter(a)\n"
            "result = max(freq.values())\n"
            "print(result)\n"
        ),
    },
    {
        "name":  "level4_sort_and_binary_search",
        "stdin": "6\n5 3 8 1 9 2\n3\n",
        "code": (
            "import sys\n"
            "import bisect\n"
            "from collections import defaultdict\n"
            "input = sys.stdin.readline\n"
            "\n"
            "n = int(input())\n"
            "a = sorted(map(int, input().split()))\n"
            "q = int(input())\n"
            "for _ in range(q):\n"
            "    pass\n"
            "print(a[-1])\n"
        ),
    },
]


def run_program(prog, run_idx):
    vm = boot_vm(f"timing-{prog['name']}-{run_idx}", CPU_MC, MEMORY_MB)
    if vm is None:
        return None
    result = run_code_on_vm(vm, prog["code"], prog["stdin"], TIMEOUT_MS)
    stop_vm(vm)
    return result


results = {}

for prog in PROGRAMS:
    print(f"\n{'='*60}")
    print(f"Program: {prog['name']}")
    print(f"{'='*60}")
    times = []
    for i in range(RUNS):
        r = run_program(prog, i)
        if r is None:
            print(f"  run {i+1}: VM boot failed")
            continue
        wt = r.get("wall_time_ms", 0)
        ec = r.get("exit_code")
        times.append(wt)
        print(f"  run {i+1}: wall={wt}ms  exit={ec}  stdout={repr(r.get('stdout','').strip()[:30])}")

    if times:
        results[prog["name"]] = {
            "runs":   times,
            "min":    min(times),
            "max":    max(times),
            "mean":   round(statistics.mean(times), 1),
            "median": round(statistics.median(times), 1),
            "p90":    round(sorted(times)[int(len(times) * 0.9)], 1),
        }
        print(f"  → min={min(times)}ms  mean={round(statistics.mean(times),1)}ms  max={max(times)}ms")
    else:
        results[prog["name"]] = {"error": "all runs failed"}

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'program':<35} {'min':>6} {'mean':>6} {'max':>6} {'p90':>6}")
print("-" * 60)
for name, s in results.items():
    if "error" in s:
        print(f"{name:<35}  FAILED")
    else:
        print(f"{name:<35} {s['min']:>6} {s['mean']:>6} {s['max']:>6} {s['p90']:>6}")

max_overhead = max(
    (s["max"] for s in results.values() if "max" in s),
    default=0
)
print(f"\nRecommended STARTUP_OVERHEAD_MS = {max_overhead}ms")
print("(use this as the floor in the reward's min_viable_tms)")

out_path = Path(__file__).parent / "timing_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
