import math
from online_rl.config import REWARD_CONFIG, MEMORY_BINS, CPU_BINS, TIMEOUT_BINS, FIXED_OVERHEAD_MB, TIMEOUT_STARTUP_OVERHEAD_MS


# def compute_llm_reward(tests_passed, llm_tier: str, llm_model: str) -> float:
#     if tests_passed is True:
#         r_quality = 1.0
#     elif tests_passed is False:
#         r_quality = -1.5
#     else:
#         r_quality = 0.5

#     r_cost = REWARD_CONFIG["llm_costs"].get(llm_tier, 0.0)

#     if llm_model == "ref-fallback":
#         r_failure = -1.2
#     elif llm_model.endswith("-escalated"):
#         r_failure = -0.5
#     else:
#         r_failure = 0.0

#     return round(r_quality + r_cost + r_failure, 4)

def compute_llm_reward(tests_passed, llm_tier: str, llm_model: str) -> float:
    # code correct
    if tests_passed is True:
        r_quality = 1.0
    # code ran but wrong answer
    elif tests_passed is False:
        r_quality = -1.5
    # code never completed (timeout, OOM, crash)
    # None means execution did not finish — not a success
    else:
        r_quality = -1

    r_cost = REWARD_CONFIG["llm_costs"].get(llm_tier, 0.0)

    if llm_model == "ref-fallback":
        r_failure = -1.2
    elif llm_model.endswith("-escalated"):
        r_failure = -0.5
    else:
        r_failure = 0.0

    return round(r_quality + r_cost + r_failure, 4)

def bins_above_optimal(chosen_val, optimal_val, bins):
    chosen_idx  = next(
        (i for i, b in enumerate(bins) if b >= chosen_val),
        len(bins) - 1
    )
    optimal_idx = next(
        (i for i, b in enumerate(bins) if b >= optimal_val),
        len(bins) - 1
    )
    return max(0, chosen_idx - optimal_idx)


def compute_res_reward(execution: dict, action: dict) -> float:

    if execution.get("exit_code") == -1:
        return -4.0

    r_timeout = -5.0 if execution.get("timed_out") else 0.0

    oom = (execution.get("oom_killed") or
           (execution.get("exit_code") == -9 and
            not execution.get("timed_out")))
    r_oom = -3.0 if oom else 0.0

    wall_ms     = max(execution.get("wall_time_ms", 1), 1)
    cpu_ms      = (execution.get("cpu_user_ms", 0) +
                   execution.get("cpu_sys_ms", 0))
    mem_peak_kb = execution.get("mem_peak_kb", 0) or 0

    # memory waste — how many bins above minimum viable did agent choose
    mem_peak_mb   = mem_peak_kb / 1024
    min_viable_mb = FIXED_OVERHEAD_MB + mem_peak_mb
    mem_bins_away = bins_above_optimal(
        action["memory_mb"], min_viable_mb, MEMORY_BINS
    )
    r_mem = -0.3 * mem_bins_away

    # timeout waste — bins above actual wall time with safety margin
    # FIX 2: floor min viable timeout at 400ms to account for Python
    # interpreter startup overhead (50-177ms inside the guest VM).
    # wall_ms already includes startup, so wall_ms * 2.0 gives a safety
    # margin. max() ensures we never reward a timeout below 400ms as viable.
    min_viable_tms = max(TIMEOUT_STARTUP_OVERHEAD_MS, wall_ms * 2.0)
    tms_bins_away  = bins_above_optimal(
        action["timeout_ms"], min_viable_tms, TIMEOUT_BINS
    )
    r_tms = -0.2 * tms_bins_away

    # cpu waste — bins above minimum viable cpu
    if wall_ms > 0 and cpu_ms > 0:
        min_viable_mc = (cpu_ms / wall_ms) * 1000 * 1.5
    else:
        min_viable_mc = 50
    cpu_bins_away = bins_above_optimal(
        action["cpu_millicores"], min_viable_mc, CPU_BINS
    )
    r_cpu = -0.2 * cpu_bins_away

    r_latency = -0.1 * math.log(wall_ms / 200 + 1)

    # efficiency bonus — reward tight allocation
    total_bins_away = mem_bins_away + tms_bins_away + cpu_bins_away
    if total_bins_away == 0:
        r_bonus = 1.0
    elif total_bins_away == 1:
        r_bonus = 0.5
    elif total_bins_away == 2:
        r_bonus = 0.2
    else:
        r_bonus = 0.0

    return round(
        r_timeout + r_oom +
        r_mem + r_tms + r_cpu +
        r_latency + r_bonus,
        4
    )