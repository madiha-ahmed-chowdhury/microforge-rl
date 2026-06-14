# Firecracker RL — Data Collection Pipeline

## What This Is

An offline RL data collection pipeline for learning **optimal resource allocation** for microVM-isolated code execution.

The core idea: given a coding task, an LLM generates code. A Firecracker microVM executes that code. The RL controller learns *which resource allocation* (CPU, memory, timeout) to assign to each VM so that execution succeeds with minimal waste.

---

## Architecture

```
                        ┌──────────────────────────────┐
                        │         Host Machine          │
                        │                               │
  EffiBench (live ───►  │  1. Load problem from HF      │
  from HuggingFace)     │                               │
                        │  2. Run test_case_generator   │
                        │     in prep VM → stress input │
                        │                               │
                        │  3. Call LLM ────────────────►│──► Claude Sonnet (hard)
                        │     ◄── generated Python code │    Claude Haiku  (easy)
                        │                               │
                        │  4. For each of 14 configs:   │
                        │     ┌─────────────────────┐   │
                        │     │  Fresh Firecracker   │   │
                        │     │  microVM             │   │
                        │     │  (e.g. 50mc / 32MB)  │   │
                        │     │  execute code        │   │
                        │     │  ← metrics           │   │
                        │     └─────────────────────┘   │
                        │     ... (14 configs total)     │
                        │                               │
                        │  5. Compute reward             │
                        │  6. Save transition            │
                        └──────────────────────────────┘
                                      │
                          collected/effibench_transitions.jsonl
                          collected/effibench_code_pairs.jsonl
                                      │
                                      ▼
                            Offline RL Training
                            (CQL discrete + TD3+BC continuous
                             + Behavioral Cloning + baselines)
```

---

## The RL Problem

| RL Component | What it is here |
|---|---|
| **State** | Code features (complexity, line count) + prompt features + recent execution history |
| **Action** | Resource allocation: CPU millicores + memory MB + timeout (continuous 3D vector) |
| **Reward** | +success, +correctness, −resource waste, −latency, −timeout penalty |
| **Next state** | Updated rolling averages of success rate, CPU used, memory used |

The goal: learn a policy that picks the *minimum* resources needed for a task to succeed — don't over-provision (wasteful) or under-provision (causes failures).

---

## Action Space (14 discrete configs)

| Config | CPU | Memory | Timeout | Use case |
|--------|-----|--------|---------|----------|
| 0  | 50mc  | 32MB  | 1s   | Absolute minimum |
| 1  | 50mc  | 48MB  | 2s   | |
| 2  | 50mc  | 64MB  | 3s   | |
| 3  | 80mc  | 48MB  | 1.5s | |
| 4  | 80mc  | 64MB  | 2.5s | |
| 5  | 80mc  | 96MB  | 4s   | |
| 6  | 100mc | 64MB  | 2s   | |
| 7  | 100mc | 96MB  | 3.5s | |
| 8  | 100mc | 128MB | 5s   | |
| 9  | 150mc | 64MB  | 3s   | |
| 10 | 150mc | 96MB  | 5s   | |
| 11 | 250mc | 128MB | 5s   | Comfortable for most tasks |
| 12 | 250mc | 192MB | 8s   | |
| 13 | 500mc | 256MB | 10s  | Maximum |

These configs are used for **data collection only**. Trained models (CQL discrete / TD3+BC continuous) learn from outcomes across all 14 and generalise to pick optimal allocations.

---

## LLM Routing

| Stress level | Model | Rationale |
|---|---|---|
| high | Claude Sonnet 4.5 | Complex algorithms need best model |
| medium | Claude Sonnet 4.5 | Reliable generation |
| low | Claude Haiku 4.5 | Cheaper, sufficient for simple problems |

Stress level is detected from problem description keywords (graph, dp, binary search, etc.).

---

## Reward Function

```
r = r_success + r_timeout + r_resource + r_latency + r_correctness

r_success     = +1.0  if exit_code == 0  else -2.0
r_timeout     = -5.0  if timed_out       else  0.0
r_resource    = -0.75 * cpu_waste - 0.75 * mem_waste   (max -1.5)
r_latency     = -0.2  * log(wall_time_ms / 500 + 1)
r_correctness = +1.0  if tests_passed    else -1.0 if tests_failed else 0.0
```

Reward is **recomputed at training time** from raw execution metrics — do not need to recollect data to change reward shaping.

---

## File Structure

```
firecracker-rl/
│
├── llm.py                  # LLM routing: Sonnet/Haiku via Anthropic SDK
├── vm.py                   # Firecracker VM lifecycle: boot, stop, vsock, cgroups
├── actions.py              # RL logic: action configs, reward, state, dataset loading
│
├── collecting_dataset.py   # CLI: stratified collection with prep VM stress input generation
├── test_one.py             # CLI: test a single random instance end-to-end
├── server.py               # FastAPI REST server wrapping all of the above
├── test_with_generator.py  # Test a single hard EffiBench problem across all 14 configs
│
├── collected/
│   ├── effibench_transitions.jsonl   # OUTPUT: (s, a, r, s', done) — 14 rows per instance
│   └── effibench_code_pairs.jsonl    # OUTPUT: prompt + generated code + reference code
│
├── dataset_tasks/          # EffiBench/security task definitions (EffiBench loaded live from HF)
├── datasets/               # MBPP and HumanEval arrow files
├── vmlinux                 # Guest kernel image
├── rootfs.ext4             # Guest root filesystem
├── firecracker             # Firecracker binary
└── .env                    # API keys (ANTHROPIC / ClAUDE_CONSOLE_API_KEY / OPENROUTER_API_KEY)
```

---

## Data Flow (per EffiBench instance)

```
1. Load problem live from HuggingFace (EffiBench/effibench-x)
   └── description, test_case_generator, canonical_solution, stress_level

2. Run test_case_generator in prep VM (500mc/256MB)
   └── tries 5 seeds, picks largest input → stress_input
       run canonical solution → expected_output (oracle)
       fallback: use pre-stored generated_tests if generator fails
       skip entirely if no usable test input found

3. Call LLM (Sonnet for hard/medium, Haiku for easy)
   └── prompt + "Return only Python code."
       → generated_code

4. Static analysis of generated code
   └── line_count, cyclomatic_complexity, has_recursion, etc.
       → state vector (14 features)

5. For each of 14 resource configs (fresh VM each):
   a. Boot Firecracker microVM with cgroups limits
   b. Send generated code + stdin mock via vsock
   c. Collect: exit_code, wall_time_ms, cpu_user_ms, mem_peak_kb
   d. Compare output to oracle → tests_passed
   e. Compute reward
   f. Tear down VM

6. Save to collected/effibench_transitions.jsonl (14 rows)
         and collected/effibench_code_pairs.jsonl  (1 row)
```

---

## Transition Format

### `collected/effibench_transitions.jsonl`
```json
{
  "episode_id": "ep_1748000000_effi_0042_3",
  "task_id": "effi_0042",
  "dataset": "effibench",
  "step": 0,
  "done": true,
  "state": {
    "prompt_token_count": 180,
    "line_count": 12,
    "cyclomatic_complexity": 4,
    "...": "14 features total"
  },
  "action": {
    "cpu_millicores": 100,
    "memory_limit_mb": 96,
    "timeout_ms": 3500
  },
  "execution": {
    "exit_code": 0,
    "timed_out": false,
    "wall_time_ms": 210,
    "cpu_user_ms": 180,
    "cpu_sys_ms": 12,
    "mem_peak_kb": 18432,
    "tests_passed": true
  },
  "reward": 1.84,
  "reward_components": {
    "r_success": 1.0, "r_timeout": 0.0,
    "r_resource": -0.12, "r_latency": -0.04, "r_correctness": 1.0
  },
  "next_state": { "...": "14 features" },
  "meta": {
    "code_hash": "a3f9c12b",
    "llm_model": "claude-sonnet-4-5",
    "collected_at": "2026-05-27T..."
  }
}
```

### `collected/effibench_code_pairs.jsonl`
```json
{
  "task_id": "effi_0042",
  "dataset": "effibench",
  "prompt": "...",
  "generated_code": "...",
  "reference_code": "...",
  "tests_passed": true,
  "llm_model": "claude-sonnet-4-5",
  "prompt_tokens": 520,
  "completion_tokens": 310,
  "llm_latency_ms": 1840,
  "collected_at": "2026-05-27T..."
}
```

---

## Running

### Prerequisites
```bash
venv/bin/pip install pyarrow fastapi uvicorn anthropic python-dotenv datasets

# .env must contain:
ClAUDE_CONSOLE_API_KEY=sk-ant-...   # Anthropic Console key (Sonnet + Haiku)
OPENROUTER_API_KEY=sk-or-...        # OpenRouter key (Laguna warmup test)
```

### Sanity check — 3 problems (1 high + 1 medium + 1 low)
```bash
venv/bin/python collecting_dataset.py --test
```

### Full collection — stratified 300 problems (30% hard / 20% medium / 50% easy)
```bash
venv/bin/python collecting_dataset.py --stratified --n 300
```

### Specific dataset / range
```bash
venv/bin/python collecting_dataset.py --dataset mbpp --n 100
venv/bin/python collecting_dataset.py --dataset effibench_large --n 100 --offset 50
```

### REST server
```bash
venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000

curl -X POST http://localhost:8000/run/one \
  -H "Content-Type: application/json" \
  -d '{"dataset": "effibench"}'

curl http://localhost:8000/status
```

---

## Why Fresh VM Per Execution?

Each of the 14 resource configs gets a **brand-new Firecracker VM** booted from scratch:

- No filesystem state leaks between executions
- No memory artifacts from previous code runs
- Resource limits are set at VM creation, not patched on a running VM
- Training data accurately reflects production (each request gets its own isolated VM)

This is slower (~14 VM boots per instance) but produces higher-quality training data.

---

## Training Plan (offline RL)

Target dataset: **300 problems × 14 configs = 4200 transitions**

Models to train and compare:

| Model | Type | Notes |
|---|---|---|
| Always-max | Baseline | Always picks 500mc/256MB/10s |
| Always-min | Baseline | Always picks 50mc/32MB/1s |
| Random | Baseline | Uniform random config |
| Rule-based | Baseline | Heuristic on code complexity features |
| Behavioral Cloning | Supervised | Learns best config per task type |
| CQL | Offline RL (discrete) | Treats 14 configs as discrete actions |
| TD3+BC | Offline RL (continuous) | Treats action as 3D continuous vector |

Reward is recomputed at training time from raw execution metrics.

---

## Next Steps

1. Collect 4200 transitions via `collecting_dataset.py --stratified --n 300`
2. Write `train.py` — load JSONL, recompute reward, train all 7 models
3. Evaluate on 55 held-out problems (300 collected + 55 held out from 355 total)
4. **Agentic loop** — rule-based reactive policy (retry with adjusted config on failure)
5. Multi-step trajectory collection + learned agentic policy (future work)

---

## Changelog

### 2026-05-27

**`actions.py`**
- EffiBench now loads live from HuggingFace (`EffiBench/effibench-x`) instead of pre-fetched JSON files — always uses full up-to-date dataset (623 problems, ~355 with Python solutions)
- Added `stress_level`, `test_case_generator`, `canonical_solution` fields to every EffiBench instance
- `effibench_large` now filters to high/medium stress problems only (no low)
- `get_output_paths` routes both `effibench` and `effibench_large` to `collected/effibench_transitions.jsonl`
- Expanded action space from 4 configs to **14 configs** (50mc–500mc, 32MB–256MB, 1s–10s)
- Cleaned transition format: removed `generated_code`, `stdout_snippet`, `stderr_snippet`, duplicate meta fields, and action index fields — transitions are now pure `(s, a, r, s', done)`
- LLM perf data (`prompt_tokens`, `completion_tokens`, `llm_latency_ms`) moved to `code_pairs` only
- Resource waste penalty in reward function increased from `0.3` to `0.75` (max `-1.5`) to strengthen the efficiency signal

**`llm.py`**
- Added `HAIKU_MODEL = "claude-haiku-4-5-20251001"`
- Routing: `high/medium stress → Claude Sonnet`, `low stress → Claude Haiku`
- `generate_code_claude` accepts a `model` parameter
- Removed Laguna as primary model for any stress level (too unreliable for EffiBench prompts)

**`collecting_dataset.py`**
- Added `_generate_stress_input()`: boots a prep VM, runs `test_case_generator` with 5 different seeds, picks the largest input to maximally stress VMs, runs canonical solution to get oracle output
- Generator timeout increased to 30s to handle slow generators
- Fallback: if generator fails, pre-stored `generated_tests` test cases are used
- Skip: if both generator and pre-stored test cases have no usable input, problem is skipped entirely (no LLM call, no VM boots wasted)
- Added `--test` flag: runs 1 high + 1 medium + 1 low problem as a sanity check
- Added `--stratified` flag: samples 30% high / 20% medium / 50% low from full EffiBench

**`server.py`**
- Added `_generate_stress_input` call before `process_instance` for EffiBench problems
- Stress input generation now consistent between CLI and REST API collection

**`test_with_generator.py`**
- Cleaned transition format to match `actions.py` (removed `generated_code`, snippets, duplicate meta)
- `meta.llm_model` now correctly records `CLAUDE_MODEL` instead of `MINIMAX_MODEL`

---

### 2026-05-27 (continued — stress input improvements)

**How reference output works:**
The reference output (oracle) is not pre-stored. For each problem:
1. Generator runs in prep VM with monkey-patched `random.randint` → produces large `stdin_data`
2. Canonical solution (from EffiBench) runs in the same prep VM with that `stdin_data` → stdout captured as `ref_output`
3. `instance["test_cases"] = [{"input": stdin_data, "output": ref_output}]`
4. LLM's generated code runs in each of the 14 config VMs with the same `stdin_data` — output compared to `ref_output` → `tests_passed`

**`collecting_dataset.py` — stress input improvements:**
- Monkey-patched `random.randint` and `random.randrange` inside the generator VM to always return the maximum value when range ≥ 100 — forces generators that do `n = random.randint(1, 100000)` to produce `n = 100000` (maximum-constraint inputs)
- Small ranges (< 100) left unchanged so structural parameters (graph flags, choice counts) still work correctly
- Increased to 7 seeds × 3 cases each = 21 candidates, picking the largest input
- Generator timeout increased to 45s to handle large input generation
- Result: wall time increased from ~10ms to ~800–1500ms on passing configs, OOM kills on 3–4 of the smallest configs per problem — meaningful resource differentiation

**`dataset_tasks/demo_effibench_structure.json`** (new file)
- Complete annotated example of one EffiBench row showing all fields: `description`, `canonical_solution_python`, `test_case_generator`, `evaluator`, `sample_generated_tests`, `time_limit_ms`, `memory_limit_mb`
- The `evaluator` field (custom checker for problems with non-unique outputs) is present in EffiBench but not yet used — currently using strict string equality. To be integrated in future.

**`actions.py`**
- Fixed `meta.llm_model` in transitions — now correctly reflects actual model used (`claude-sonnet-4-5` for high/medium, `claude-haiku-4-5-20251001` for low) instead of always showing `MINIMAX_MODEL`

---

### 2026-06-01 — Full data collection completed

**Dataset collected**
- **278 problems × 14 configs = 3892 transitions** saved to `collected/effibench_transitions.jsonl`
- Stratified sample: 30% high / 20% medium / 50% low stress, fixed seed=42 for reproducibility
- ~22% of attempted problems were skipped (generator failed + no pre-stored test cases)
- All 278 problems have exactly 14 transitions — no incomplete or partial entries

**Data quality findings**
- 48% of transitions have `tests_passed=True`, 28% wrong code, 24% timed out / crashed
- cfg00 (50mc/32MB/1s): 100% timeout — strong "avoid" signal for RL
- cfg02 (50mc/64MB/3s): first config where most problems succeed — clear threshold signal
- Memory usage clusters tightly at ~13MB across all problems — memory dimension has weak RL signal
- Wall time drops from ~2000ms at cfg01 to ~15ms at cfg13 — CPU/timeout is the main differentiator
- 89% of problems have cfg10 (150mc/96MB/5s) as optimal config by reward — RL learns efficiency over max provisioning
- Mean reward: −1.2 across all transitions (expected — many low-resource configs fail)

**`actions.py`**
- Added `stress_level` to `meta` in every transition for training-time slicing

**`collecting_dataset.py`**
- Added `--skip-existing` flag: reads the output file and skips any `task_id` already present — prevents duplicates when topping up from a larger stratified pool
- Crash recovery: `send_code` call inside `_generate_stress_input` wrapped in try/except — prep VM crashes now fall back to pre-stored test cases instead of killing the entire run
- `--offset` now works correctly with `--stratified`: slices the stratified list after sampling

**Collection issues encountered and resolved**
- Crash at problem 84/300: prep VM returned empty vsock response → `json.loads("")` raised `JSONDecodeError` → killed entire run. Fixed by wrapping `send_code` in try/except.
- Duplicate transitions from top-up run: `--n 400 --offset 300` generated a different stratified sample than `--n 300`, causing 24 problems to appear in both. Fixed by de-duplicating (keep first 14 per task_id) and adding `--skip-existing` flag.
- `stress_level` missing from meta: first 83 problems collected before the fix show `stress_level=unknown` in meta. Not critical — derivable from problem text at training time.

---

### 2026-06-10 — Online RL: code caching + problem pool expansion

**`online_rl/code_cache.py`** (new file)
- Persistent JSON cache mapping `task_id → {code, model_used, passed, tests_passed, cached_at, use_count}`
- `should_refresh()` returns True only when `tests_passed is False` (wrong code) — resource failures are SAC's problem, not the cache's
- `increment_use()` saves to disk every 50 hits to avoid write overhead on hot problems
- `stats()` prints `cached=N passed=N avg_use=N.N` for quick health checks

**`online_rl/runner.py`**
- LLM calls now happen **once per problem**, not once per episode — on a cache hit the PREP VM is skipped entirely and `llm_reward = 0.0` (no LLM decision, no cost)
- Cache miss path: calls LLM → runs PREP VM → stores result → updates bandit, identical to before
- CONFIG VM (Agent 2 / resource allocator) still runs **every episode** regardless of cache hit — SAC continues learning resource allocation with cached code
- Added `--refresh-cache` flag: deletes `code_cache.json` and regenerates all codes from scratch
- Startup prints: `total problems in problem cache`, `problems already in code cache`, `problems needing first LLM call` — shows exactly how much LLM work remains
- Cache stats printed every 100 episodes alongside bandit checkpoint

**`online_rl/config.py`**
- Added `"code_cache": "online_rl/results/code_cache.json"` to `PATHS`

**`online_rl/prepare_inputs.py`**
- Added `--add-new N` flag: streams the full `deepmind/code_contests` HuggingFace dataset, finds problems not already in `cc_pool_cache.json` by `task_id`, samples N randomly (time-seeded so each run picks different problems), and appends them to the pool
- Added `--min-rating` / `--max-rating` filters for `--add-new` (default 1400–1800)
- New problems get `"test_case_generator": ""` — generators can be added later via `generate_stress_tests.py`
- Usage: `python -m online_rl.prepare_inputs --add-new 50`

**LLM call reduction**
- Before: ~10 000 LLM calls for 10 000 episodes over 300 problems
- After: ~300 calls (one per problem on first encounter), then ~0
- Problems with failing code (`tests_passed=False`) are automatically re-attempted next episode

---

## Running Online RL

```bash
python3 online_rl/runner.py [options]
```

| Flag | Default | Explanation |
|---|---|---|
| `--episodes N` | 3000 | Number of training episodes to run |
| `--pool N` | 150 | How many problems to use from `cc_pool_cache.json`. Takes the first N problems (slice, not random). Raise this to use more of the 394 available. |
| `--min-rating N` | 1400 | Lower bound on Codeforces rating when filtering the pool |
| `--max-rating N` | 1800 | Upper bound on Codeforces rating |
| `--resume` | off | Resume from the latest checkpoint in `online_rl/checkpoints/` |
| `--refresh-cache` | off | Delete `code_cache.json` and regenerate all LLM code from scratch |
| `--task PARTIAL_ID` | off | Run only the problem whose `task_id` contains this string (enables verbose output) |
| `--force-cpu N` | off | Override Agent 2 CPU decision with a fixed value (millicores) |
| `--force-mem N` | off | Override Agent 2 memory decision with a fixed value (MB) |
| `--force-timeout N` | off | Override Agent 2 timeout decision with a fixed value (ms) |
| `--dry-run` | off | Skip LLM and VM calls entirely — uses ref solution + synthetic execution for fast testing |

### Examples

```bash
# Basic run — 3000 episodes, 150 problems
python3 online_rl/runner.py

# Use all 394 problems in the pool
python3 online_rl/runner.py --episodes 5000 --pool 394

# Resume from last checkpoint
python3 online_rl/runner.py --resume --episodes 2000

# Debug a specific problem with verbose output
python3 online_rl/runner.py --task cc_1234A --episodes 1

# Force fixed resources (bypass Agent 2) to isolate LLM bandit training
python3 online_rl/runner.py --force-cpu 200 --force-mem 128 --force-timeout 5000

# Add more problems to the pool before running
python3 -m online_rl.prepare_inputs --add-new 50
python3 online_rl/runner.py --pool 444
```

Episodes:
ep 600-695 breakdown:

ep 600-619: -2.316, 1/20 positive, 5 failures
ep 620-639: -2.538, 1/20 positive, 6 failures
ep 640-659: -1.972, 3/20 positive, 2 failures
ep 660-679: -1.869, 2/20 positive, 2 failures
ep 680-699: -2.248, 0/20 positive, 2 failures
The positive rate has dropped significantly from the 5-6/20 we were seeing earlier.

Block	Avg	Positive
ep 0-99 (warmup)	-2.490	5/100
ep 100-199 (warmup)	-2.380	6/100
ep 200-299 (early train)	-2.307	8/100
ep 300-399	-1.682	12/100
ep 400-499 (entropy↓)	-1.973	16/100
ep 500-599	-1.884	17/100

after chanhes
config.py

Reduced target entropy from 98% to 70% of maximum at ep 400 to shift from exploration toward exploitation after the reward plateau
Lowered all learning rates by 3x (actor/alpha: 3e-4→1e-4, critic: 5e-4→2e-4) to stabilise training at later episodes
sac_agent.py

Save now includes optimizer states so momentum is preserved across restarts, not just network weights
Load accepts a config to overwrite learning rates after restoring momentum — keeps Adam's accumulated gradients but uses the new lower rates
runner.py

--checkpoint flag to resume from a specific checkpoint file, with episode counter continuing from the number in the filename and matching buffer/bandit/scaler loaded automatically
Entropy annealing: after ep 500, alpha decays by 0.5% every 10 episodes, down to a floor of 0.01, so the policy gradually commits rather than staying at fixed entropy
Best model tracking: saves sac_best.pt whenever the 50-episode rolling average reward improves, so the best policy seen during training is always preserved
supervisor_update.md

Progress report covering the two datasets, the two-agent design (bandit for LLM selection, discrete SAC for resource allocation), methodology diagram, reward function breakdown, and the learning curve from 600 episodes

Block	Avg	Positive	Failures
ep 560-579	-2.218	1/20	6 (cold start)
ep 580-599	-2.130	1/20	3
ep 600-619	-2.170	1/20	2
ep 620-639	-1.905	2/20	1
ep 640-659	-1.808	4/20	4 (peak)
ep 660-679	-1.964	1/20	4
ep 680-699	-2.323	0/20	3
ep 700-719	-1.838	3/20	2
ep 720-739	-1.996	1/20	3
ep 740-759	-2.112	1/20	2
ep 760-779	-2.595	0/20	4
ep 780-799	-2.088	1/20	2
ep 800-819	-1.972	3/20	3
ep 820-839	-1.639	4/20	2
ep 840-859	-1.580	2/20	4
