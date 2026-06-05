import random
import time

from online_rl.config import PREP_CONFIG


def boot_vm(vm_id: str, cpu_millicores: int, memory_mb: int):
    from vm import start_vm, wait_for_agent, apply_cgroups, stop_vm
    vm = start_vm(vm_id)
    time.sleep(3)
    if not wait_for_agent(vm):
        stop_vm(vm)
        return None
    apply_cgroups(vm, cpu_millicores, max(128, memory_mb))
    return vm


def run_code_on_vm(vm, code: str, stdin: str, timeout_ms: int) -> dict:
    from vm import send_code
    wrapped = f"import sys, io\nsys.stdin = io.StringIO({repr(stdin)})\n{code}"
    try:
        return send_code(vm, wrapped, timeout_ms)
    except Exception as e:
        return {
            "exit_code": -1, "timed_out": False, "wall_time_ms": 0,
            "cpu_user_ms": 0, "cpu_sys_ms": 0, "mem_peak_kb": 0,
            "stdout": "", "stderr": str(e),
        }


def dry_run_execution(action: dict) -> dict:
    wall_ms = random.randint(200, action["timeout_ms"] // 2)
    cpu_ms  = int(wall_ms * (action["cpu_millicores"] / 1000) * random.uniform(0.3, 0.9))
    mem_kb  = int(action["memory_mb"] * 1024 * random.uniform(0.2, 0.7))
    return {
        "exit_code":    0,
        "timed_out":    False,
        "wall_time_ms": wall_ms,
        "cpu_user_ms":  cpu_ms,
        "cpu_sys_ms":   0,
        "mem_peak_kb":  mem_kb,
        "stdout":       "",
        "stderr":       "",
    }
