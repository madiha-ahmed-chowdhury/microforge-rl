import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from online_rl.config import (
    FEATURE_COLS, PATHS, PREP_CONFIG,
    MAX_RESOURCE_RETRIES,
    LLM_SAC_CONFIG, RES_SAC_CONFIG,
    LLM_TIERS,
)
from online_rl.replay_buffer  import ReplayBuffer
from online_rl.sac_agent      import SACAgent
from online_rl.problem_loader import load_cc_problems
from online_rl.state_builder  import static_analyse, build_state_vec, update_rolling, scale_action, select_llm_tier
from online_rl.rewards        import compute_llm_reward, compute_res_reward
from online_rl.vm_runner      import boot_vm, run_code_on_vm, dry_run_execution
from online_rl.llm_caller     import generate_code, run_prep_vm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",   type=int, default=3000)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--dry-run",    action="store_true", dest="dry_run")
    parser.add_argument("--min-rating", type=int, default=1400)
    parser.add_argument("--max-rating", type=int, default=1800)
    parser.add_argument("--pool",       type=int, default=150)
    args = parser.parse_args()

    print(f"[runner] streaming CC problems | rating={args.min_rating}-{args.max_rating} | pool={args.pool}")
    problems = load_cc_problems(args.min_rating, args.max_rating, args.pool)
    print(f"[runner] pool ready: {len(problems)} problems")

    scaler = None
    if os.path.exists(PATHS["scaler"]):
        with open(PATHS["scaler"], "rb") as f:
            scaler = pickle.load(f)
        print(f"[runner] scaler loaded")
    else:
        print(f"[runner] WARNING: no scaler at {PATHS['scaler']}")

    llm_agent  = SACAgent(LLM_SAC_CONFIG)
    res_agent  = SACAgent(RES_SAC_CONFIG)
    llm_buffer = ReplayBuffer(LLM_SAC_CONFIG["buffer_size"])
    res_buffer = ReplayBuffer(RES_SAC_CONFIG["buffer_size"])

    start_ep = 0
    if args.resume:
        ckpt_dir      = Path(PATHS["checkpoints"])
        llm_ckpts     = sorted(ckpt_dir.glob("llm_ep_*.pt"))
        res_ckpts     = sorted(ckpt_dir.glob("res_ep_*.pt"))
        llm_buf_ckpts = sorted(ckpt_dir.glob("llm_buf_*.pkl"))
        res_buf_ckpts = sorted(ckpt_dir.glob("res_buf_*.pkl"))
        if llm_ckpts and res_ckpts:
            llm_agent.load(str(llm_ckpts[-1]))
            res_agent.load(str(res_ckpts[-1]))
            start_ep = int(llm_ckpts[-1].stem.split("_")[2])
            print(f"[runner] agents resumed from ep {start_ep}")
        if llm_buf_ckpts and res_buf_ckpts:
            with open(llm_buf_ckpts[-1], "rb") as f:
                llm_buffer = pickle.load(f)
            with open(res_buf_ckpts[-1], "rb") as f:
                res_buffer = pickle.load(f)
            print(f"[runner] buffers restored: llm={len(llm_buffer)} res={len(res_buffer)}")

    os.makedirs(PATHS["checkpoints"], exist_ok=True)
    os.makedirs(PATHS["results"],     exist_ok=True)

    reward_window   = []
    llm_tier_counts = {t: 0 for t in LLM_TIERS}
    opus_used       = 0

    for ep in range(start_ep, start_ep + args.episodes):
        problem      = random.choice(problems)
        task_id      = problem["task_id"]
        description  = problem["description"]
        stdin        = problem["stdin"]
        expected     = problem["expected"]
        ref_solution = problem["ref_solution"]

        from actions import build_state as _build_state
        base_state    = _build_state(description, static_analyse(ref_solution), task_type=3)
        llm_state_vec = build_state_vec(base_state, {}, scaler)

        llm_action_raw = llm_agent.select_action(llm_state_vec)
        llm_tier       = select_llm_tier(float(llm_action_raw[0]), ep, opus_used, ep - start_ep + 1)
        llm_tier_counts[llm_tier] += 1
        if llm_tier == "opus":
            opus_used += 1

        if args.dry_run:
            code               = ref_solution
            llm_model          = "dry-run"
            llm_tests_passed   = None
            res_state_vec      = build_state_vec(base_state, static_analyse(code), scaler)
            res_action_raw     = res_agent.select_action(res_state_vec)
            current_state_vec  = res_state_vec
            current_action_raw = res_action_raw
            final_action       = scale_action(res_action_raw)
            final_exec         = dry_run_execution(final_action)
        else:
            print(f"[runner] ep {ep:04d} | llm={llm_tier} | task={task_id}")
            code, llm_model = generate_code(description, stdin, expected, llm_tier, ref_solution)

            # PREP VM: verify + refine LLM code at max config
            prep_vm = boot_vm(f"prep-{ep}", PREP_CONFIG["cpu_millicores"], PREP_CONFIG["memory_mb"])
            if prep_vm is None:
                print(f"[runner] PREP VM failed — skipping ep {ep}")
                continue
            from vm import stop_vm as _stop_vm
            code, llm_tests_passed = run_prep_vm(prep_vm, code, stdin, expected, description, run_code_on_vm)
            _stop_vm(prep_vm)
            print(f"[runner] PREP done | correct={'✓' if llm_tests_passed else '✗'}")

            # Agent 2: select resources based on generated code features
            res_state_vec  = build_state_vec(base_state, static_analyse(code), scaler)
            res_action_raw = res_agent.select_action(res_state_vec)
            final_action   = scale_action(res_action_raw)

            # CONFIG VM: run code under resource constraints
            rolling_state     = list(res_state_vec)
            current_state_vec = list(res_state_vec)
            current_action_raw = res_action_raw
            final_exec        = None
            config_vm         = None

            for attempt in range(MAX_RESOURCE_RETRIES):
                config_vm = boot_vm(f"cfg-{ep}-{attempt}", final_action["cpu_millicores"], final_action["memory_mb"])
                if config_vm is None:
                    break

                exec_result = run_code_on_vm(config_vm, code, stdin,
                                             final_action["timeout_ms"],
                                             memory_limit_mb=final_action["memory_mb"])
                exec_result["tests_passed"] = (
                    exec_result.get("exit_code") == 0 and bool(expected) and
                    exec_result.get("stdout", "").strip() == expected.strip()
                ) if exec_result.get("exit_code") == 0 and expected else None

                if exec_result.get("timed_out"):
                    # store this attempt with its real timeout penalty
                    attempt_reward   = compute_res_reward(exec_result, final_action)
                    next_rolling     = update_rolling(current_state_vec, exec_result)
                    res_buffer.push(current_state_vec, current_action_raw, attempt_reward, next_rolling, False)

                    rolling_state[FEATURE_COLS.index("recent_success_rate")] *= 0.9
                    current_state_vec  = list(rolling_state)
                    current_action_raw = res_agent.select_action(current_state_vec)
                    final_action       = scale_action(current_action_raw)
                    _stop_vm(config_vm)
                    config_vm = None
                    continue

                if exec_result.get("exit_code") == -9 and not exec_result.get("timed_out"):
                    # store this attempt with its real OOM penalty
                    attempt_reward   = compute_res_reward(exec_result, final_action)
                    next_rolling     = update_rolling(current_state_vec, exec_result)
                    res_buffer.push(current_state_vec, current_action_raw, attempt_reward, next_rolling, False)

                    rolling_state[FEATURE_COLS.index("recent_mean_mem_used")] *= 1.2
                    current_state_vec  = list(rolling_state)
                    current_action_raw = res_agent.select_action(current_state_vec)
                    final_action       = scale_action(current_action_raw)
                    _stop_vm(config_vm)
                    config_vm = None
                    continue

                final_exec = exec_result
                _stop_vm(config_vm)
                config_vm = None
                break

            if config_vm is not None:
                _stop_vm(config_vm)
            if final_exec is None:
                final_exec = {
                    "exit_code": -1, "timed_out": False, "wall_time_ms": 0,
                    "cpu_user_ms": 0, "cpu_sys_ms": 0, "mem_peak_kb": 0,
                    "stdout": "", "stderr": "vm boot failed", "tests_passed": None,
                }

        # ── Train Agent 1 ─────────────────────────────────────────────────────
        llm_reward   = compute_llm_reward(llm_tests_passed, llm_tier, llm_model)
        llm_next_vec = update_rolling(llm_state_vec, final_exec)
        llm_buffer.push(llm_state_vec, llm_action_raw, llm_reward, llm_next_vec, True)
        if llm_buffer.is_ready(LLM_SAC_CONFIG["warmup"]):
            llm_agent.update(llm_buffer.sample(LLM_SAC_CONFIG["batch_size"]))

        # ── Train Agent 2 ─────────────────────────────────────────────────────
        res_reward   = compute_res_reward(final_exec, final_action)
        res_next_vec = update_rolling(current_state_vec, final_exec)
        res_buffer.push(current_state_vec, current_action_raw, res_reward, res_next_vec, True)
        if res_buffer.is_ready(RES_SAC_CONFIG["warmup"]):
            res_agent.update(res_buffer.sample(RES_SAC_CONFIG["batch_size"]))

        # ── Logging ───────────────────────────────────────────────────────────
        reward_window.append(res_reward)
        if len(reward_window) > 10:
            reward_window.pop(0)
        avg10 = round(sum(reward_window) / len(reward_window), 3)

        transition = {
            "episode":      ep,
            "task_id":      task_id,
            "cf_rating":    problem["cf_rating"],
            "cf_tags":      problem["cf_tags"],
            "llm_tier":     llm_tier,
            "llm_model":    llm_model,
            "llm_reward":   llm_reward,
            "action":       final_action,
            "execution": {
                "exit_code":    final_exec.get("exit_code"),
                "timed_out":    final_exec.get("timed_out"),
                "wall_time_ms": final_exec.get("wall_time_ms"),
                "cpu_user_ms":  final_exec.get("cpu_user_ms"),
                "mem_peak_kb":  final_exec.get("mem_peak_kb"),
                "tests_passed": final_exec.get("tests_passed"),
            },
            "res_reward":   res_reward,
            "avg10":        avg10,
            "bufs":         [len(llm_buffer), len(res_buffer)],
        }
        with open(PATHS["transitions"], "a") as f:
            f.write(json.dumps(transition) + "\n")

        last_ep = ep == start_ep + args.episodes - 1
        if (ep + 1) % 10 == 0 or last_ep:
            tp      = final_exec.get("tests_passed")
            correct = "✓" if tp is True else ("✗" if tp is False else "?")
            sign    = "+" if res_reward >= 0 else ""
            print(
                f"ep {ep+1:04d} | {task_id} cf={problem['cf_rating']} | llm={llm_tier}({llm_model}) | "
                f"cpu={final_action['cpu_millicores']}mc mem={final_action['memory_mb']}MB "
                f"t={final_action['timeout_ms']//1000}s | "
                f"exit={final_exec.get('exit_code')} wall={final_exec.get('wall_time_ms')}ms "
                f"correct={correct} oom={final_exec.get('oom_killed', False)} | "
                f"llm_r={'+' if llm_reward>=0 else ''}{llm_reward:.2f} "
                f"res_r={sign}{res_reward:.2f} avg10={'+' if avg10>=0 else ''}{avg10:.2f} | "
                f"bufs={len(llm_buffer)}/{len(res_buffer)} tiers={llm_tier_counts}"
            )
            with open(PATHS["curve"], "a") as f:
                f.write(json.dumps({"ep": ep+1, "llm_r": llm_reward, "res_r": res_reward,
                                    "avg10": avg10, "llm_tier": llm_tier}) + "\n")

        if (ep + 1) % 10 == 0:
            ck = PATHS["checkpoints"]
            llm_agent.save(f"{ck}llm_ep_{ep+1:05d}.pt")
            res_agent.save(f"{ck}res_ep_{ep+1:05d}.pt")
            with open(f"{ck}llm_buf_{ep+1:05d}.pkl", "wb") as f:
                pickle.dump(llm_buffer, f)
            with open(f"{ck}res_buf_{ep+1:05d}.pkl", "wb") as f:
                pickle.dump(res_buffer, f)
            print(f"[runner] checkpoint saved at ep {ep+1}")

    print(f"[runner] done. {args.episodes} episodes.")


if __name__ == "__main__":
    main()
