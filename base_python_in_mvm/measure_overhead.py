import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from online_rl.vm_runner import boot_vm, run_code_on_vm
from vm import stop_vm

# code that reports guest memory stats
diagnostic_code = """
import subprocess, os

# total and free RAM in guest
meminfo = open('/proc/meminfo').read()
lines = {l.split(':')[0]: int(l.split()[1]) for l in meminfo.strip().splitlines()}
total_kb  = lines['MemTotal']
free_kb   = lines['MemFree']
avail_kb  = lines['MemAvailable']
used_kb   = total_kb - avail_kb

print(f"guest_total_kb={total_kb}")
print(f"guest_used_kb={used_kb}")
print(f"guest_free_kb={free_kb}")
print(f"guest_avail_kb={avail_kb}")

# agent.py process memory
for line in open('/proc/self/status').readlines():
    if line.startswith('VmRSS') or line.startswith('VmPeak'):
        print(f"this_process_{line.strip()}")
"""

for memory_mb in [64, 128]:
    print(f"\n--- {memory_mb}MB allocation ---")
    vm = boot_vm(f"diag-{memory_mb}", cpu_millicores=100, memory_mb=memory_mb)
    if vm is None:
        print("VM boot failed")
        continue

    result = run_code_on_vm(vm, diagnostic_code, stdin="", timeout_ms=10000, memory_limit_mb=0)
    stop_vm(vm)

    print(result.get("stdout", ""))
    if result.get("stderr"):
        print("stderr:", result.get("stderr"))
