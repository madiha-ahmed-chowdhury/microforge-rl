import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from online_rl.vm_runner import boot_vm, run_code_on_vm
from vm import stop_vm

code = open("base_python_in_mvm/program.py").read()

results = []

for memory_mb in [32, 64, 128]:
    print(f"\n--- testing with {memory_mb}MB ---")
    vm = boot_vm(f"base-test-{memory_mb}", cpu_millicores=100, memory_mb=memory_mb)
    if vm is None:
        print(f"VM boot failed at {memory_mb}MB")
        results.append({"memory_mb": memory_mb, "error": "vm boot failed"})
        continue

    result = run_code_on_vm(vm, code, stdin="", timeout_ms=10000, memory_limit_mb=memory_mb)
    stop_vm(vm)

    entry = {
        "memory_mb":    memory_mb,
        "exit_code":    result.get("exit_code"),
        "stdout":       result.get("stdout", "").strip(),
        "stderr":       result.get("stderr", "").strip(),
        "wall_time_ms": result.get("wall_time_ms"),
        "cpu_user_ms":  result.get("cpu_user_ms"),
        "mem_peak_kb":  result.get("mem_peak_kb"),
        "mem_peak_mb":  round(result.get("mem_peak_kb", 0) / 1024, 2),
        "timed_out":    result.get("timed_out"),
        "oom_killed":   result.get("oom_killed"),
    }
    results.append(entry)
    print(json.dumps(entry, indent=2))

print("\n=== summary ===")
for r in results:
    if "error" in r:
        print(f"  {r['memory_mb']}MB: FAILED ({r['error']})")
    else:
        status = "✓" if r["exit_code"] == 0 else "✗ OOM" if r.get("oom_killed") else "✗"
        print(f"  {r['memory_mb']}MB alloc | peak={r['mem_peak_mb']}MB | wall={r['wall_time_ms']}ms | {status}")

with open("base_python_in_mvm/result.json", "w") as f:
    json.dump({"program": code, "results": results}, f, indent=2)
print("\nsaved to base_python_in_mvm/result.json")
