import math
from online_rl.config import REWARD_CONFIG


def compute_llm_reward(tests_passed, llm_tier: str, llm_model: str) -> float:
    if tests_passed is True:
        r_quality = 1.0
    elif tests_passed is False:
        r_quality = -1.5
    else:
        r_quality = 0.5

    r_cost = REWARD_CONFIG["llm_costs"].get(llm_tier, 0.0)

    if llm_model == "ref-fallback":
        r_failure = -1.2
    elif llm_model.endswith("-escalated"):
        r_failure = -0.5
    else:
        r_failure = 0.0

    return round(r_quality + r_cost + r_failure, 4)


def compute_res_reward(execution: dict, action: dict) -> float:
    rc = REWARD_CONFIG
    if execution.get("timed_out"):
        return rc["r_timeout"]
    # guest-side OOM (ulimit -v exceeded, exit 137) or host-side OOM (exit -9)
    if execution.get("oom_killed") or (execution.get("exit_code") == -9 and not execution.get("timed_out")):
        return rc["r_oom"]

    wall_ms       = max(execution.get("wall_time_ms", 1), 1)
    cpu_ms        = execution.get("cpu_user_ms", 0) + execution.get("cpu_sys_ms", 0)
    wall_sec      = wall_ms / 1000
    cpu_budget_ms = (action["cpu_millicores"] / 1000) * wall_sec * 1000
    cpu_waste     = max(0, (cpu_budget_ms - cpu_ms) / max(cpu_budget_ms, 1))
    mem_alloc_kb  = action["memory_mb"] * 1024
    mem_waste     = max(0, (mem_alloc_kb - execution.get("mem_peak_kb", 0)) / max(mem_alloc_kb, 1))

    r_resource = rc["cpu_waste_w"] * cpu_waste + rc["mem_waste_w"] * mem_waste
    r_latency  = rc["latency_w"] * math.log(wall_ms / rc["latency_base"] + 1)
    r_success  = 1.0 if execution.get("exit_code") == 0 else -2.0

    return round(r_success + r_resource + r_latency, 4)
