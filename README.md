# Firecracker RL — Two-Agent Online RL for Microvm Resource Allocation

## What This Is

An **online RL system** that learns two things simultaneously:

1. **Agent 1 (LLM Bandit)** — which LLM to call for a given coding problem (Thompson sampling over model tiers)
2. **Agent 2 (Discrete SAC)** — which CPU / memory / timeout bin to allocate to the Firecracker microVM that runs the generated code

The core loop: pick a problem → bandit selects a model → LLM generates Python code → PREP VM verifies/refines it → SAC picks resource bin → CONFIG VM executes → rewards computed → both agents update.

---

## Architecture

```
Problem from pool
        │
        ▼
┌───────────────────┐
│  Agent 1: Bandit  │  Thompson sampling over {free, sonnet, opus}
│  (LLM selection)  │  → generates Python code
└────────┬──────────┘
         │  code
         ▼
┌───────────────────┐
│  PREP VM          │  500 CPU / 512 MB / 60s  (fixed)
│  (verify+refine)  │  runs code, checks tests, optionally refines
└────────┬──────────┘
         │  verified code
         ▼
┌───────────────────┐
│  Agent 2: SAC     │  Discrete SAC over (cpu_bin × mem_bin × timeout_bin)
│  (resource alloc) │  state = 9 code+history features
└────────┬──────────┘
         │  resource config
         ▼
┌───────────────────┐
│  CONFIG VM        │  Firecracker microVM booted with SAC-chosen limits
│  (execution)      │  → exit_code, wall_time_ms, mem_peak_kb, tests_passed
└───────────────────┘
         │
         ▼
   Reward computed → both agents updated
   Transition logged to results/transitions.jsonl
```

---

## Agents

### Agent 1 — LLM Bandit

- **Algorithm**: Thompson sampling (Beta prior, Bernoulli reward)
- **Arms**: `free` (gpt-oss-120b), `sonnet` (claude-sonnet-4-6), `opus` (claude-opus-4-8)
- **Reward signal**: `llm_reward` — +1.0 for correct code, penalised for cost tier and fallback
- **Code cache**: generated code is cached per `task_id`; on a cache hit the bandit is skipped (no LLM call, no cost)

### Agent 2 — Discrete SAC

- **Algorithm**: Discrete Soft Actor-Critic
- **Action space**: `cpu_idx` (9 bins) × `mem_idx` (8 bins) × `timeout_idx` (15 bins)
- **State vector** (9 features): line count, cyclomatic complexity, has_recursion, has_sort, rolling avg wall time, rolling avg mem, rolling success rate, CF rating (normalised), CF tag embedding
- **Reward signal**: `res_reward` — penalises over-allocation and failure; bonus for tight fit
- **Exploration**: entropy annealing after episode 500 (alpha decays 0.5% per 10 episodes, floor 0.01)

---

## Reward Functions

### `llm_reward`
```
+1.0   tests passed on PREP VM
-1.0   tests failed
-0.5   model cost penalty for sonnet tier
-1.5   model cost penalty for opus tier
-0.5   ref-solution fallback used
```

### `res_reward`
```
base = -2.0 if OOM/crash
       -1.0 if timed out
       +0.5 if tests passed, else -0.5

efficiency = -(wasted_cpu_ratio + wasted_mem_ratio) * 0.5

+1.0 bonus if allocation within 20% of actual usage (tight fit)
```

---

## Training Pool

- **Source**: `deepmind/code_contests` (HuggingFace)
- **Size**: 394 problems in `online_rl/cc_pool_cache.json`
- **Filters**: must have Python 3 reference solution + generated test inputs
- **Rating range**: 800–2800 Codeforces

---

## Training Results (1060 episodes)

| Block | Avg res_reward | Notes |
|---|---|---|
| ep 0–99 (warmup) | -2.490 | random exploration |
| ep 100–199 | -2.380 | |
| ep 200–299 | -2.307 | early SAC learning |
| ep 300–399 | -1.682 | significant improvement |
| ep 400–499 | -1.973 | entropy reduced to 70% |
| ep 500–599 | -1.884 | |
| ep 820–839 | -1.639 | near-peak |
| ep 840–859 | -1.580 | best 20-ep block |

Training was stopped at **1060 episodes** after the reward plateau. Best checkpoint saved as `online_rl/checkpoints/sac_best.pt`.

---

## Evaluation

### Eval Set

60 new problems never seen during training, from `deepmind/code_contests`, split into four categories:

| Category | n | CF rating | CF tags |
|---|---|---|---|
| high_memory | 20 | any | graphs / trees / dfs / dp / data structures |
| easy | 10 | 800–1199 | any |
| medium | 20 | 1200–1599 | any |
| hard | 10 | 1600+ | any |

Stored in `online_rl/cc_pool_cache_test.json`. Each problem has a hand-crafted stress-test generator verified against the reference solution (60/60 passing).

### Eval Protocol

- **Agent 1**: bypassed — reference solution used directly as code (`--use-ref`)
- **Agent 2**: deterministic (greedy argmax, no exploration, no SAC updates)
- **No code cache** — every problem gets fresh execution
- Logged to `online_rl/results/eval_transitions.jsonl`

```bash
python3 -m online_rl.runner \
  --checkpoint online_rl/checkpoints/sac_best.pt \
  --eval \
  --use-ref \
  --pool-cache online_rl/cc_pool_cache_test.json \
  --episodes 60
```

### Eval Results — Agent 2 (SAC / Resource)

| Category | n | pass% | avg res_reward | timeouts |
|---|---|---|---|---|
| high_memory | 17 | 41.2% | -3.082 | 0 |
| easy | 9 | 66.7% | -0.716 | 0 |
| medium | 25 | 72.0% | -1.532 | 0 |
| hard | 9 | 66.7% | -0.346 | 0 |
| **OVERALL** | **60** | **61.7%** | **-1.671** | **0** |

**Training avg**: -1.7 → **Eval avg**: -1.671 — essentially identical, indicating good generalisation.

**Key observations:**
- `high_memory` is the weakest category — graph/tree/DP problems have less predictable memory footprints and the SAC agent over-allocates or misses
- `hard` (1600+ rating) achieves the best res_reward (-0.346) despite being the highest-rated — reference solutions for hard problems tend to be clean and predictable, making resource allocation easier
- **Zero timeouts** across all 60 eval episodes — SAC never under-allocates time

### Resource Bin Distribution (eval)

```
CPU idx : {0:2, 1:7, 2:3, 3:3, 4:7, 5:4, 6:6, 7:1, 8:27}   ← biased toward high CPU
Mem idx : {0:4, 1:12, 2:35, 3:2, 4:4, 6:1, 7:2}             ← clusters around mid mem
```

The SAC agent strongly prefers the highest CPU bin (idx 8, 27/60 episodes) — it learned during training that CPU is the main bottleneck for passing tests.

---

## File Structure

```
firecracker-rl/
│
├── online_rl/
│   ├── runner.py                  # Main training loop (both agents)
│   ├── config.py                  # Hyperparameters, paths, bin definitions
│   ├── sac_agent.py               # Discrete SAC (Agent 2)
│   ├── bandit.py                  # Thompson sampling bandit (Agent 1)
│   ├── rewards.py                 # llm_reward + res_reward functions
│   ├── llm_caller.py              # LLM routing (free/sonnet/opus)
│   ├── code_cache.py              # Per-task LLM code cache
│   ├── problem_loader.py          # Loads cc_pool_cache.json
│   ├── prepare_inputs.py          # Add problems to pool from HuggingFace
│   │
│   ├── fetch_eval_problems.py     # Fetches 60 unseen eval problems
│   ├── generate_stress_tests_eval.py  # Stress generators for eval problems (60/60)
│   ├── eval_results.py            # Print RL1+RL2 tables from eval_transitions.jsonl
│   │
│   ├── cc_pool_cache.json         # 394 training problems
│   ├── cc_pool_cache_test.json    # 60 eval problems with embedded generators
│   │
│   ├── checkpoints/
│   │   ├── sac_best.pt            # Best SAC checkpoint (saved on rolling avg improvement)
│   │   ├── res_ep_NNNNN.pt        # SAC checkpoint every 10 episodes
│   │   └── res_buf_NNNNN.pkl      # Replay buffer checkpoint
│   │
│   └── results/
│       ├── transitions.jsonl      # Training transitions (s, a, r, s', done)
│       ├── curve.jsonl            # Per-episode reward curve
│       ├── code_cache.json        # Cached LLM-generated code per task_id
│       └── eval_transitions.jsonl # Eval run transitions (60 episodes)
│
├── vmlinux                        # Guest kernel image
├── rootfs.ext4                    # Guest root filesystem
└── firecracker                    # Firecracker binary
```

---

## Running

### Prerequisites
```bash
venv/bin/pip install anthropic datasets torch numpy
# .env must contain: ANTHROPIC_API_KEY, OPENROUTER_API_KEY
```

### Training
```bash
# Basic run — 3000 episodes, all 394 problems
python3 -m online_rl.runner --episodes 3000

# Resume from last checkpoint
python3 -m online_rl.runner --resume --episodes 1000

# Resume from specific checkpoint
python3 -m online_rl.runner --checkpoint online_rl/checkpoints/sac_best.pt --episodes 500

# Debug a single problem (verbose)
python3 -m online_rl.runner --task cc_1234A --episodes 1

# Force fixed resources (bypass SAC, isolate bandit training)
python3 -m online_rl.runner --force-cpu 200 --force-mem 128 --force-timeout 5000
```

### Evaluation — full two-agent eval (costs LLM credits)
```bash
python3 -m online_rl.runner \
  --checkpoint online_rl/checkpoints/sac_best.pt \
  --eval \
  --no-code-cache \
  --pool-cache online_rl/cc_pool_cache_test.json \
  --episodes 60
```

### Evaluation — Agent 2 only, no LLM calls (free)
```bash
python3 -m online_rl.runner \
  --checkpoint online_rl/checkpoints/sac_best.pt \
  --eval \
  --use-ref \
  --pool-cache online_rl/cc_pool_cache_test.json \
  --episodes 60

python3 online_rl/eval_results.py
```

### Expand the training pool
```bash
python3 -m online_rl.prepare_inputs --add-new 50
python3 -m online_rl.runner --episodes 2000
```

### Rebuild eval generators (if problems change)
```bash
python3 online_rl/generate_stress_tests_eval.py   # verify + embed → cc_pool_cache_test.json
```

---

## Runner Flags

| Flag | Default | Description |
|---|---|---|
| `--episodes N` | 3000 | Number of episodes |
| `--pool N` | all | Problems to use from pool cache |
| `--resume` | off | Resume from latest checkpoint |
| `--checkpoint PATH` | off | Resume from specific `.pt` file |
| `--eval` | off | Deterministic SAC, no updates, log to eval_transitions.jsonl |
| `--no-code-cache` | off | Skip code cache — always call LLM fresh |
| `--use-ref` | off | Use reference solution as code — skip LLM entirely (free eval) |
| `--pool-cache PATH` | cc_pool_cache.json | Alternate problem pool (e.g. eval set) |
| `--dry-run` | off | Synthetic execution — no LLM, no VMs |
| `--force-cpu N` | off | Override SAC CPU decision |
| `--force-mem N` | off | Override SAC memory decision |
| `--force-timeout N` | off | Override SAC timeout decision |
| `--task ID` | off | Run only one problem (enables verbose) |

---

## Changelog

### 2026-06-16 — Eval pipeline + 60-problem test set

**`online_rl/fetch_eval_problems.py`** (new)
- Streams `deepmind/code_contests`, skips all 394 training problems, fills 4 buckets: 20 high-memory (graph/tree/dp/data-structure tags), 10 easy (800–1199), 20 medium (1200–1599), 10 hard (1600+)
- Saves to `cc_pool_cache_test.json` — same format as training pool

**`online_rl/generate_stress_tests_eval.py`** (new)
- Hand-crafted stress generators for all 60 eval problems
- `verify()`: runs each generator, pipes output to ref solution, embeds passing generators as `test_case_generator` field in `cc_pool_cache_test.json`
- All 60/60 generators verified and embedded

**`online_rl/runner.py`**
- Added `--pool-cache PATH` — load alternate problem pool instead of `cc_pool_cache.json`
- Added `--no-code-cache` — always call LLM fresh (skip cache lookup)
- Added `--use-ref` — use `ref_solution` directly as code, skip bandit + PREP VM entirely (zero API cost, pure Agent 2 eval)

**`online_rl/problem_loader.py`**
- `load_cc_problems()` now accepts optional `cache_path` parameter

**`online_rl/eval_results.py`** (new)
- Reads `eval_transitions.jsonl`, prints RL1 (bandit/model distribution) and RL2 (res_reward by category, bin distribution) result tables

---

### 2026-06-10 — Online RL: code caching + problem pool expansion

**`online_rl/code_cache.py`** (new)
- Persistent JSON cache mapping `task_id → {code, model_used, passed, use_count}`
- `should_refresh()` returns True only when `tests_passed is False`
- LLM calls reduced from ~10 000 (one per episode) to ~394 (one per unique problem)

**`online_rl/runner.py`**
- Cache hit: skip PREP VM, `llm_reward = 0.0`, SAC still runs
- Added `--refresh-cache` flag

**`online_rl/prepare_inputs.py`**
- `--add-new N` flag: finds N new problems from HuggingFace not already in pool, appends to `cc_pool_cache.json`
- `--min-rating` / `--max-rating` filters

---

### 2026-06-01 — Entropy annealing + best model tracking

**`online_rl/config.py`**
- Target entropy reduced from 98% to 70% of maximum at episode 400 (shift from exploration to exploitation)
- All learning rates reduced 3× at episode 400 (actor/alpha: 3e-4→1e-4, critic: 5e-4→2e-4)

**`online_rl/sac_agent.py`**
- Checkpoint now saves optimizer states (Adam momentum preserved across restarts)
- `load()` accepts config to override learning rates while keeping accumulated gradients

**`online_rl/runner.py`**
- `--checkpoint` flag to resume from a specific file
- Entropy annealing: after episode 500, alpha decays 0.5% per 10 episodes, floor 0.01
- Best model tracking: saves `sac_best.pt` whenever 50-episode rolling average improves

---

### 2026-05-27 — Offline data collection (v1)

Initial system: offline RL data collection pipeline using EffiBench problems, 14 fixed resource configs, 278 problems × 14 configs = 3892 transitions. Replaced in June 2026 by the online two-agent system.
