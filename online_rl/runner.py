import argparse
import json
import math
import os
import torch
import pickle
import random
import subprocess
import sys
import tempfile
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
    SAC_FEATURE_COLS, PATHS, PREP_CONFIG,
    MAX_RESOURCE_RETRIES,
    RES_SAC_CONFIG, DQN_CONFIG, PPO_CONFIG,
    CPU_BINS, MEMORY_BINS, TIMEOUT_BINS,
)
from online_rl.replay_buffer         import DiscreteReplayBuffer
from online_rl.sac_agent             import DiscreteSACAgent
from online_rl.sac_agent_factored    import DiscreteSACAgentFactored
from online_rl.dqn_agent             import DQNAgent
from online_rl.ppo_agent             import PPOAgent
from online_rl.llm_bandit            import LLMBandit


def _nearest_idx(value: float, bins: list) -> int:
    return min(range(len(bins)), key=lambda i: abs(bins[i] - value))
from online_rl.problem_loader import load_cc_problems
from online_rl.state_builder  import static_analyse, build_state_vec, update_rolling
from online_rl.rewards        import compute_llm_reward, compute_res_reward
from online_rl.vm_runner      import boot_vm, run_code_on_vm, dry_run_execution
from vm import stop_vm as _stop_vm
from online_rl.llm_caller     import generate_code, run_prep_vm
from online_rl.code_cache     import CodeCache


def _run_script(code: str, stdin: str = "", timeout: int = 60):
    """Run a Python script, return (stdout, returncode)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(["python3", path], input=stdin.encode(),
                           capture_output=True, timeout=timeout)
        return r.stdout.decode(errors='replace'), r.returncode
    except subprocess.TimeoutExpired:
        return "", 1
    except Exception:
        return "", 1
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",     type=int, default=3000)
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--dry-run",      action="store_true", dest="dry_run")
    # parser.add_argument("--min-rating",   type=int, default=1400)
    # parser.add_argument("--max-rating",   type=int, default=1800)
    # parser.add_argument("--pool",         type=int, default=150)
    parser.add_argument("--task",         type=str, default=None,
                        help="Run only this problem (partial task_id match)")
    parser.add_argument("--force-cpu",    type=int, default=None,
                        help="Override Agent 2 CPU millicores")
    parser.add_argument("--force-mem",    type=int, default=None,
                        help="Override Agent 2 memory MB")
    parser.add_argument("--force-timeout",type=int, default=None,
                        help="Override Agent 2 timeout ms")
    parser.add_argument("--refresh-cache", action="store_true", dest="refresh_cache",
                        help="Clear the code cache and regenerate all codes from scratch")
    parser.add_argument("--checkpoint",    type=str, default=None,
                        help="Load a specific res_ep_NNNNN.pt checkpoint and continue from that episode")
    parser.add_argument("--eval",          action="store_true",
                        help="Evaluation mode: deterministic actions, no SAC updates, logs to eval_transitions.jsonl")
    parser.add_argument("--eval-problems", type=str, default=None,
                        help="Path to JSON file with list of task_ids to use for evaluation")
    parser.add_argument("--pool-cache",    type=str, default=None,
                        help="Path to alternate pool cache JSON (default: cc_pool_cache.json)")
    parser.add_argument("--no-code-cache", action="store_true", dest="no_code_cache",
                        help="Skip code cache — always call LLM fresh (for evaluation)")
    parser.add_argument("--use-ref", action="store_true", dest="use_ref",
                        help="Skip LLM entirely — use ref_solution as code (for RL2-only eval)")
    parser.add_argument("--agent", choices=["shared", "factored", "dqn", "ppo"], default="shared",
                        help="Agent architecture: shared, factored, dqn, or ppo")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Named run: transitions saved to results/transitions_{run_name}.jsonl, "
                             "checkpoints to checkpoints/{run_name}/")
    args = parser.parse_args()

    _run_tag          = args.run_name if args.run_name else args.agent
    _ckpt_dir         = f"online_rl/checkpoints/{_run_tag}/"
    _transitions_path = f"online_rl/results/transitions_{_run_tag}.jsonl" if args.run_name \
                        else (PATHS["transitions"] if args.agent == "shared" else f"online_rl/results/transitions_{args.agent}.jsonl")
    _agent_cfg        = DQN_CONFIG if args.agent == "dqn" else (PPO_CONFIG if args.agent == "ppo" else RES_SAC_CONFIG)
    verbose = args.task is not None

    print(f"[runner] loading all problems from pool cache")
    problems = load_cc_problems(cache_path=args.pool_cache)
    print(f"[runner] pool ready: {len(problems)} problems")

    if args.eval_problems:
        with open(args.eval_problems) as f:
            eval_ids = set(json.load(f))
        problems = [p for p in problems if p["task_id"] in eval_ids]
        print(f"[runner] eval mode: filtered to {len(problems)} problems from {args.eval_problems}")
    elif args.eval:
        print(f"[runner] eval mode: using full problem pool (no --eval-problems specified)")

    from sklearn.preprocessing import StandardScaler
    scaler        = StandardScaler()
    scaler_fitted = False
    scaler_buffer = []

    if args.agent == "factored":
        res_agent = DiscreteSACAgentFactored(_agent_cfg)
    elif args.agent == "dqn":
        res_agent = DQNAgent(_agent_cfg)
    elif args.agent == "ppo":
        res_agent = PPOAgent(_agent_cfg)
    else:
        res_agent = DiscreteSACAgent(_agent_cfg)
    res_buffer = DiscreteReplayBuffer(_agent_cfg["buffer_size"]) if args.agent != "ppo" else None
    bandit     = LLMBandit()

    start_ep = 0
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        try:
            res_agent.load(str(ckpt_path), cfg=_agent_cfg)
            start_ep = int(ckpt_path.stem.split("_")[-1])
            print(f"[runner] agent loaded from {ckpt_path.name}, continuing from ep {start_ep}")
        except Exception as e:
            print(f"[runner] WARNING: could not load checkpoint ({e}) — starting fresh weights")
        if args.agent != "ppo":
            buf_path = ckpt_path.parent / ckpt_path.name.replace("res_ep_", "res_buf_").replace(".pt", ".pkl")
            if buf_path.exists():
                try:
                    with open(buf_path, "rb") as f:
                        res_buffer = pickle.load(f)
                    print(f"[runner] buffer restored: {len(res_buffer)} transitions")
                except Exception as e:
                    print(f"[runner] WARNING: could not load buffer ({e})")
        ep_str = ckpt_path.stem.split("_")[-1]
        bandit_path = ckpt_path.parent / f"bandit_ep_{ep_str}.json"
        if bandit_path.exists():
            bandit.load(str(bandit_path))
            print(f"[runner] bandit resumed from {bandit_path.name}")
        if os.path.exists(PATHS["online_scaler"]):
            with open(PATHS["online_scaler"], "rb") as f:
                scaler = pickle.load(f)
            scaler_fitted = True
            print(f"[runner] loaded online scaler from {PATHS['online_scaler']}")
    elif args.resume:
        ckpt_dir      = Path(_ckpt_dir)
        res_ckpts     = sorted(ckpt_dir.glob("res_ep_*.pt"))
        res_buf_ckpts = sorted(ckpt_dir.glob("res_buf_*.pkl"))
        if res_ckpts:
            try:
                res_agent.load(str(res_ckpts[-1]), cfg=_agent_cfg)
                start_ep = int(res_ckpts[-1].stem.split("_")[2])
                print(f"[runner] agent resumed from ep {start_ep}")
            except Exception as e:
                print(f"[runner] WARNING: could not load agent checkpoint ({e}) — starting fresh weights")
        if res_buf_ckpts and args.agent != "ppo":
            try:
                with open(res_buf_ckpts[-1], "rb") as f:
                    res_buffer = pickle.load(f)
                print(f"[runner] buffer restored: res={len(res_buffer)}")
            except Exception as e:
                print(f"[runner] WARNING: could not load replay buffer ({e}) — starting empty")
        bandit_ckpts = sorted(ckpt_dir.glob("bandit_ep_*.json"))
        if bandit_ckpts:
            bandit.load(str(bandit_ckpts[-1]))
            print(f"[runner] bandit resumed from {bandit_ckpts[-1].name}")
        if os.path.exists(PATHS["online_scaler"]):
            with open(PATHS["online_scaler"], "rb") as f:
                scaler = pickle.load(f)
            scaler_fitted = True
            print(f"[runner] loaded online scaler from {PATHS['online_scaler']}")

    os.makedirs(_ckpt_dir, exist_ok=True)
    os.makedirs(PATHS["results"],     exist_ok=True)

    if args.refresh_cache:
        if os.path.exists(PATHS["code_cache"]):
            os.remove(PATHS["code_cache"])
        print("[runner] cache cleared")
    cache = CodeCache(PATHS["code_cache"])
    print(f"[runner] code cache: {cache.stats()}")
    print(f"[runner] total problems in problem cache: {len(problems)}")
    print(f"[runner] problems already in code cache: {len([t for t in problems if cache.has(t['task_id'])])}")
    print(f"[runner] problems needing first LLM call: {len([t for t in problems if not cache.has(t['task_id'])])}")

    rolling = {
        "recent_success_rate":  0.5,
        "recent_mean_cpu_used": 0.0,
        "recent_mean_mem_used": 0.0,
    }
    reward_window   = []
    reward_window50 = []
    best_avg50      = -float("inf")

    for ep in range(start_ep, start_ep + args.episodes):
        if args.task:
            match = [p for p in problems if args.task in p["task_id"]]
            if not match:
                print(f"[runner] ERROR: no problem matching '{args.task}'")
                break
            problem = match[0]
        else:
            problem = random.choice(problems)

        task_id      = problem["task_id"]
        description  = problem["description"]
        stdin        = problem["stdin"]
        expected     = problem["expected"]
        ref_solution = problem["ref_solution"]

        if problem.get("test_case_generator"):
            if verbose:
                print(f"\n[verbose] Running test case generator for {task_id}...")
            gen_stdin, rc = _run_script(problem["test_case_generator"])
            if rc == 0 and gen_stdin.strip():
                ref_out, ref_rc = _run_script(ref_solution, stdin=gen_stdin)
                if ref_rc == 0 and ref_out.strip():
                    stdin    = gen_stdin
                    expected = ref_out
                    if verbose:
                        print(f"[verbose] Generator OK — stdin={len(stdin)} chars")
                        print(f"[verbose] stdin preview (first 5 lines):")
                        for ln in stdin.splitlines()[:5]:
                            print(f"          {ln}")
                        print(f"[verbose] expected output preview: {repr(expected[:200])}")
                else:
                    if verbose:
                        print(f"[verbose] Ref solution failed on generated input — using cached stdin")
            else:
                if verbose:
                    print(f"[verbose] Generator failed — using cached stdin")
        elif verbose:
            print(f"[verbose] No generator — using cached stdin ({len(stdin)} chars)")
            print(f"[verbose] stdin preview: {repr(stdin[:200])}")

        active_scaler = scaler if scaler_fitted else None

        if args.dry_run:
            llm_tier            = "free"
            code                = ref_solution
            llm_model           = "dry-run"
            llm_tests_passed    = None
            code_features       = static_analyse(code)
            current_state_vec   = build_state_vec(code_features, rolling, active_scaler)
            current_action_dict = res_agent.select_action(current_state_vec)
            final_action        = current_action_dict
            final_exec          = dry_run_execution(final_action)
            llm_reward          = compute_llm_reward(llm_tests_passed, llm_tier, llm_model)
        else:
            print(f"[runner] ep {ep:04d} | task={task_id}")

            if args.use_ref:
                code      = ref_solution
                llm_model = "ref"
                llm_tier  = "ref"
                llm_reward = 0.0
                # PREP VM: run ref_solution with generated stdin to get oracle expected output
                prep_vm = boot_vm(f"prep-{ep}", PREP_CONFIG["cpu_millicores"], PREP_CONFIG["memory_mb"])
                if prep_vm is None:
                    print(f"[runner] PREP VM failed — skipping ep {ep}")
                    continue
                prep_exec = run_code_on_vm(
                    prep_vm, ref_solution, stdin,
                    PREP_CONFIG["timeout_ms"],
                    memory_limit_mb=PREP_CONFIG["memory_mb"],
                )
                _stop_vm(prep_vm)
                if prep_exec.get("exit_code") == 0 and prep_exec.get("stdout", "").strip():
                    expected         = prep_exec["stdout"]
                    llm_tests_passed = True
                else:
                    llm_tests_passed = False
                    print(f"[runner] ref PREP failed exit={prep_exec.get('exit_code')} — tests_passed=False")
                print(f"[runner] ep={ep} task={task_id} stdin_len={len(stdin)} chars ref=True")
            elif not args.no_code_cache and cache.has(task_id) and not cache.should_refresh(task_id):
                cached           = cache.get(task_id)
                code             = cached["code"]
                llm_model        = cached["model_used"]
                llm_tests_passed = cached["passed"]
                llm_tier         = "cached"
                cache.increment_use(task_id)
                llm_reward       = 0.0
                print(f"[runner] using cached code for {task_id} "
                      f"(use #{cached['use_count']})")
            else:
                chosen_model, model_idx = bandit.select(problem)
                llm_tier                = chosen_model["tier"]
                code, llm_model         = generate_code(description, stdin, expected, llm_tier, ref_solution)

                used_ref_fallback = False
                if code is None:
                    code              = ref_solution
                    llm_model         = "ref-fallback"
                    used_ref_fallback = True

                # PREP VM: verify + refine LLM code at max config
                prep_vm = boot_vm(f"prep-{ep}", PREP_CONFIG["cpu_millicores"], PREP_CONFIG["memory_mb"])
                if prep_vm is None:
                    print(f"[runner] PREP VM failed — skipping ep {ep}")
                    continue
                code, llm_tests_passed, prep_result = run_prep_vm(prep_vm, code, stdin, expected, description, run_code_on_vm)
                _stop_vm(prep_vm)
                print(f"[runner] PREP done | correct={'✓' if llm_tests_passed else '✗'}")

                cache.store(
                    task_id      = task_id,
                    code         = code,
                    model_used   = llm_model,
                    passed       = llm_tests_passed,
                    tests_passed = llm_tests_passed,
                )

                bandit.update(problem, model_idx, llm_tests_passed, used_ref_fallback)
                if verbose:
                    print(f"\n[verbose] ── PREP VM ──────────────────────────────────")
                    print(f"[verbose] stdin passed to prep VM: {len(stdin)} chars")
                    print(f"[verbose] expected output: {repr(expected[:300])}")
                    print(f"[verbose] LLM tests passed: {llm_tests_passed}")
                    print(f"[verbose] PREP exit_code={prep_result.get('exit_code')}  wall={prep_result.get('wall_time_ms')}ms  mem={prep_result.get('mem_peak_kb')}KB")
                    prep_stdout = prep_result.get('stdout', '')
                    prep_stderr = prep_result.get('stderr', '')
                    if prep_stdout:
                        print(f"[verbose] PREP stdout ({len(prep_stdout)} chars): {repr(prep_stdout[:300])}")
                    if prep_stderr:
                        print(f"[verbose] PREP stderr: {repr(prep_stderr[:300])}")
                    print(f"[verbose] Generated code ({len(code)} chars):")
                    for i, ln in enumerate(code.splitlines()[:20]):
                        print(f"          {i+1:3d}: {ln}")
                    if len(code.splitlines()) > 20:
                        print(f"          ... ({len(code.splitlines())} lines total)")

                llm_reward = compute_llm_reward(llm_tests_passed, llm_tier, llm_model)

            # Agent 2: select resources based on generated code features
            code_features       = static_analyse(code)
            current_state_vec   = build_state_vec(code_features, rolling, active_scaler)
            current_action_dict = res_agent.select_action(current_state_vec, deterministic=args.eval)
            final_action        = dict(current_action_dict)

            # Override with forced values if provided
            if args.force_cpu is not None:
                final_action["cpu_millicores"] = args.force_cpu
                final_action["cpu_idx"]        = _nearest_idx(args.force_cpu, CPU_BINS)
            if args.force_mem is not None:
                final_action["memory_mb"]      = args.force_mem
                final_action["mem_idx"]        = _nearest_idx(args.force_mem, MEMORY_BINS)
            if args.force_timeout is not None:
                final_action["timeout_ms"]     = args.force_timeout
                final_action["timeout_idx"]    = _nearest_idx(args.force_timeout, TIMEOUT_BINS)
            current_action_dict = final_action

            if verbose:
                print(f"\n[verbose] ── AGENT 2 (Resource Allocator) ──────────────")
                print(f"[verbose] cpu={final_action['cpu_millicores']}mc  mem={final_action['memory_mb']}MB  timeout={final_action['timeout_ms']}ms")

            # CONFIG VM: run code under resource constraints
            attempt_rolling = dict(rolling)
            final_exec      = None
            config_vm       = None

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
                    attempt_reward  = compute_res_reward(exec_result, final_action)
                    next_rolling    = update_rolling(attempt_rolling, exec_result)
                    next_state_vec  = build_state_vec(code_features, next_rolling, active_scaler)
                    if args.agent != "ppo":
                        res_buffer.push(current_state_vec,
                                        current_action_dict["cpu_idx"],
                                        current_action_dict["mem_idx"],
                                        current_action_dict["timeout_idx"],
                                        attempt_reward, next_state_vec, False)
                    attempt_rolling["recent_success_rate"] *= 0.9
                    current_state_vec   = build_state_vec(code_features, attempt_rolling, active_scaler)
                    current_action_dict = res_agent.select_action(current_state_vec)
                    final_action        = dict(current_action_dict)
                    _stop_vm(config_vm)
                    config_vm = None
                    continue

                if exec_result.get("exit_code") == -9 and not exec_result.get("timed_out"):
                    attempt_reward  = compute_res_reward(exec_result, final_action)
                    next_rolling    = update_rolling(attempt_rolling, exec_result)
                    next_state_vec  = build_state_vec(code_features, next_rolling, active_scaler)
                    if args.agent != "ppo":
                        res_buffer.push(current_state_vec,
                                        current_action_dict["cpu_idx"],
                                        current_action_dict["mem_idx"],
                                        current_action_dict["timeout_idx"],
                                        attempt_reward, next_state_vec, False)
                    attempt_rolling["recent_mean_mem_used"] *= 1.2
                    current_state_vec   = build_state_vec(code_features, attempt_rolling, active_scaler)
                    current_action_dict = res_agent.select_action(current_state_vec)
                    final_action        = dict(current_action_dict)
                    _stop_vm(config_vm)
                    config_vm = None
                    continue

                final_exec = exec_result
                _stop_vm(config_vm)
                config_vm = None
                if verbose:
                    print(f"\n[verbose] ── CONFIG VM result ──────────────────────────")
                    print(f"[verbose] exit_code={exec_result.get('exit_code')}  timed_out={exec_result.get('timed_out')}")
                    print(f"[verbose] wall_time={exec_result.get('wall_time_ms')}ms  mem_peak={exec_result.get('mem_peak_kb')}KB")
                    print(f"[verbose] tests_passed={exec_result.get('tests_passed')}")
                    stdout = exec_result.get('stdout', '')
                    stderr = exec_result.get('stderr', '')
                    print(f"[verbose] stdout ({len(stdout)} chars): {repr(stdout[:300])}")
                    if stderr:
                        print(f"[verbose] stderr: {repr(stderr[:300])}")
                break

            if config_vm is not None:
                _stop_vm(config_vm)
            if final_exec is None:
                final_exec = {
                    "exit_code": -1, "timed_out": False, "wall_time_ms": 0,
                    "cpu_user_ms": 0, "cpu_sys_ms": 0, "mem_peak_kb": 0,
                    "stdout": "", "stderr": "vm boot failed", "tests_passed": None,
                }

        # ── Train Agent 2 (Resource Allocator) ───────────────────────────────
        res_reward    = compute_res_reward(final_exec, final_action)
        next_rolling  = update_rolling(rolling, final_exec)
        next_state_vec = build_state_vec(code_features, next_rolling, active_scaler)
        if not args.eval:
            if args.agent == "ppo":
                res_agent.store(
                    current_state_vec,
                    current_action_dict["cpu_idx"],
                    current_action_dict["mem_idx"],
                    current_action_dict["timeout_idx"],
                    current_action_dict.get("log_prob", 0.0),
                    current_action_dict.get("value", 0.0),
                    res_reward, True,
                )
                if res_agent.ready_to_update():
                    res_agent.update()
            else:
                res_buffer.push(current_state_vec,
                                current_action_dict["cpu_idx"],
                                current_action_dict["mem_idx"],
                                current_action_dict["timeout_idx"],
                                res_reward, next_state_vec, True)
                if res_buffer.is_ready(max(_agent_cfg["warmup"], _agent_cfg["batch_size"])):
                    res_agent.update(res_buffer.sample(_agent_cfg["batch_size"]))
                    if args.agent in ("shared", "factored") and ep > 500 and ep % 10 == 0:
                        with torch.no_grad():
                            res_agent.log_alpha.data = torch.clamp(
                                res_agent.log_alpha.data * 0.995,
                                min=math.log(0.01),
                            )
                    if args.agent == "dqn" and ep >= _agent_cfg["warmup"]:
                        res_agent.decay_epsilon()

        # ── Update rolling history ────────────────────────────────────────────
        rolling = next_rolling

        # ── Fit scaler after warmup ───────────────────────────────────────────
        if not scaler_fitted:
            import numpy as np
            merged = {**code_features, **rolling}
            raw = [float(merged.get(k, 0.0)) for k in SAC_FEATURE_COLS]
            scaler_buffer.append(raw)
            if len(scaler_buffer) >= 200:
                scaler.fit(np.array(scaler_buffer))
                scaler_fitted = True
                scaler_buffer.clear()
                with open(PATHS["online_scaler"], "wb") as f:
                    pickle.dump(scaler, f)
                print(f"[runner] scaler fitted and saved at ep {ep + 1}")

        # ── Logging ───────────────────────────────────────────────────────────
        reward_window.append(res_reward)
        if len(reward_window) > 10:
            reward_window.pop(0)
        avg10 = round(sum(reward_window) / len(reward_window), 3)

        reward_window50.append(res_reward)
        if len(reward_window50) > 50:
            reward_window50.pop(0)
        if len(reward_window50) == 50:
            avg50 = sum(reward_window50) / 50
            if avg50 > best_avg50:
                best_avg50 = avg50
                res_agent.save(f"{_ckpt_dir}sac_best.pt")
                print(f"[runner] new best model at ep {ep} | avg50={avg50:.3f}")

        transition = {
            "episode":    ep,
            "task_id":    task_id,
            "cf_rating":  problem["cf_rating"],
            "cf_tags":    problem["cf_tags"],
            "llm_tier":   llm_tier,
            "llm_model":  llm_model,
            "bucket":     bandit.get_bucket(problem),
            "llm_reward": llm_reward,
            "action":     final_action,
            "sac_state":  dict(zip(SAC_FEATURE_COLS, [round(float(x), 4) for x in current_state_vec])),
            "execution": {
                "exit_code":    final_exec.get("exit_code"),
                "timed_out":    final_exec.get("timed_out"),
                "wall_time_ms": final_exec.get("wall_time_ms"),
                "cpu_user_ms":  final_exec.get("cpu_user_ms"),
                "mem_peak_kb":  final_exec.get("mem_peak_kb"),
                "tests_passed": final_exec.get("tests_passed"),
            },
            "res_reward": res_reward,
            "avg10":      avg10,
            "buf":        len(res_agent._rollout) if args.agent == "ppo" else len(res_buffer),
        }
        _eval_suffix = f"_{_run_tag}"
        transitions_path = _transitions_path if not args.eval else PATHS["results"] + f"eval_transitions{_eval_suffix}.jsonl"
        with open(transitions_path, "a") as f:
            f.write(json.dumps(transition) + "\n")

        last_ep = ep == start_ep + args.episodes - 1
        if (ep + 1) % 10 == 0 or last_ep:
            tp      = final_exec.get("tests_passed")
            correct = "✓" if tp is True else ("✗" if tp is False else "?")
            sign    = "+" if res_reward >= 0 else ""
            print(
                f"ep {ep+1:04d} | {task_id} cf={problem['cf_rating']} | tier={llm_tier}({llm_model}) | "
                f"cpu={final_action['cpu_millicores']}mc mem={final_action['memory_mb']}MB "
                f"t={final_action['timeout_ms']//1000}s | "
                f"exit={final_exec.get('exit_code')} wall={final_exec.get('wall_time_ms')}ms "
                f"correct={correct} oom={final_exec.get('oom_killed', False)} | "
                f"llm_r={'+' if llm_reward>=0 else ''}{llm_reward:.2f} "
                f"res_r={sign}{res_reward:.2f} avg10={'+' if avg10>=0 else ''}{avg10:.2f} | "
                f"buf={len(res_agent._rollout) if args.agent == 'ppo' else len(res_buffer)}"
            )
            with open(PATHS["curve"], "a") as f:
                f.write(json.dumps({"ep": ep+1, "llm_r": llm_reward, "res_r": res_reward,
                                    "avg10": avg10, "llm_tier": llm_tier}) + "\n")

        if not args.eval and (ep + 1) % 10 == 0:
            ck = _ckpt_dir
            res_agent.save(f"{ck}res_ep_{ep+1:05d}.pt")
            if args.agent != "ppo":
                with open(f"{ck}res_buf_{ep+1:05d}.pkl", "wb") as f:
                    pickle.dump(res_buffer, f)
            bandit.save(f"{ck}bandit_ep_{ep+1:05d}.json")
            print(f"[runner] checkpoint saved at ep {ep+1}")
            if (ep + 1) % 100 == 0:
                print(f"[cache] {cache.stats()}")
        if (ep + 1) % 200 == 0:
            bandit.summary()

    print(f"[runner] done. {args.episodes} episodes.")


if __name__ == "__main__":
    main()
