# Firecracker RL — Two-Agent Online RL for MicroVM Resource Allocation

## System Overview

An **online reinforcement learning system** that learns to allocate compute resources for Firecracker microVMs running competitive programming solutions. The system uses a two-agent design:

1. **Agent 1 — LLM Bandit**: selects which language model to call for code generation (Thompson sampling)
2. **Agent 2 — Resource Allocator**: selects CPU, memory, and timeout limits for the Firecracker microVM that executes the code

Four variants of Agent 2 were implemented and compared: Discrete SAC with a shared trunk (SAC-Shared), Discrete SAC with factored trunks (SAC-Factored), Deep Q-Network (DQN), and Proximal Policy Optimization (PPO). All four agents use the same 9-feature state vector, the same action space, and the same reward function.

---

## Two-Agent Pipeline

```
Problem from pool (cc_pool_cache.json)
         │
         ▼
┌─────────────────────┐
│  Agent 1: Bandit    │  Thompson sampling over {free, sonnet, opus}
│  (LLM selection)    │  Reward: +1 correct code, −cost tier
└──────────┬──────────┘
           │  Python code
           ▼
┌─────────────────────┐
│  PREP VM            │  Fixed: 500 CPU / 512 MB / 60 s
│  (verify + refine)  │  Runs code against reference, optionally refines (1 attempt)
└──────────┬──────────┘
           │  verified code
           ▼
┌─────────────────────┐
│  Agent 2: Resource  │  Picks (cpu_bin, mem_bin, timeout_bin) from state
│  Allocator          │  State = 9 code + history features
└──────────┬──────────┘
           │  resource config
           ▼
┌─────────────────────┐
│  CONFIG VM          │  Firecracker microVM booted at Agent 2's chosen limits
│  (execution)        │  Returns: exit_code, wall_time_ms, mem_peak_kb, tests_passed
└─────────────────────┘
           │
           ▼
    res_reward computed → Agent 2 updated
    llm_reward computed → Bandit updated
    Transition logged to results/transitions.jsonl
```

When training Agent 2 in isolation (`--use-ref`), Agent 1 is bypassed entirely — the reference solution from the dataset is used directly as code, eliminating LLM API calls.

---

## Agent 1 — Thompson Sampling Bandit

| Property | Value |
|---|---|
| Algorithm | Thompson sampling (Beta-Bernoulli) |
| Arms | `free` (gpt-oss-120b), `sonnet` (claude-sonnet-4-6), `opus` (claude-opus-4-8) |
| Prior | Beta(1, 1) per arm — uniform uninformative prior |
| Update | Success (tests\_passed=True) → Beta(α+1, β), Failure → Beta(α, β+1) |
| Reward | +1.0 correct, −0.5 sonnet cost, −1.5 opus cost, −0.5 ref fallback |
| Code cache | Generated code cached per task\_id; cache hit skips bandit (no LLM call, llm\_reward=0) |

---

## State Space — 18 Features

Agent 2 uses an 18-dimensional state vector split into three groups: static code features (AST analysis), memory/timeout signal features (AST pattern detection), and rolling execution history (EMA updated after each episode).

Features 0–8 (v1) were used for SAC-Shared, SAC-Factored, DQN, and the first PPO run. Features 9–17 (v2/v3) were added after all four agents failed on high-memory problems.

| # | Feature | Group | Scale | Description |
|---|---|---|---|---|
| 0 | `cyclomatic_complexity` | Static | ÷50 | 1 + number of branches, loops, comprehensions in AST |
| 1 | `max_loop_depth` | Static | ÷10 | Maximum nesting depth of `for`/`while` loops |
| 2 | `estimated_complexity` | Static | ÷5 | Ordinal 0–3: O(1)→O(n)→O(n²)→O(n³+) |
| 3 | `has_recursion` | Static | ÷1 | Binary: any function name appears in its own call graph |
| 4 | `ast_node_count` | Static | ÷1000 | Total AST node count — proxy for code size |
| 5 | `line_count` | Static | ÷200 | Non-blank line count |
| 6 | `recent_success_rate` | Rolling | ÷1 | EMA(α=0.1) of pass/fail |
| 7 | `recent_mean_cpu_used` | Rolling | ÷200 | EMA of `cpu_user_ms` from CONFIG VM |
| 8 | `recent_mean_mem_used` | Rolling | ÷50000 | EMA of `mem_peak_kb` from CONFIG VM |
| 9 | `uses_defaultdict` | Memory signal | ÷1 | `from collections import defaultdict` detected |
| 10 | `uses_deque` | Memory signal | ÷1 | `from collections import deque` detected |
| 11 | `uses_heapq` | Memory signal | ÷1 | `import heapq` detected |
| 12 | `has_array_mult` | Memory signal | ÷1 | `[x]*N` pattern — segment trees, BIT arrays, DP tables |
| 13 | `has_collection_list` | Memory signal | ÷1 | `[[] for _ in range(n)]` — adjacency lists, set graphs |
| 14 | `uses_itertools` | Timeout signal | ÷1 | `import itertools` — combinatorial explosion risk |
| 15 | `has_lru_cache` | Timeout signal | ÷1 | `@lru_cache` / `@cache` decorator detected |
| 16 | `sort_call_count` | Timeout signal | ÷10 | Count of `sorted()` + `.sort()` calls |
| 17 | `has_while_true` | Timeout signal | ÷1 | `while True:` loop detected |

Rolling features start at zero and update via `new = 0.1 × measurement + 0.9 × old`. The scaler is fitted online after 200 warmup episodes using `sklearn.StandardScaler`.

**Why the memory signal features matter:** Without features 9–13, the agent cannot distinguish a 50-line graph BFS (`[[] for _ in range(n)]` → needs 256 MB) from a 50-line string problem (needs 80 MB). Both look identical on features 0–8.

---

## Action Space

Agent 2 selects three resource dimensions simultaneously. Each dimension is discretised into bins:

| Dimension | Bins | N |
|---|---|---|
| CPU (millicores) | [50, 75, 100, 125, 150, 175, 200, 300, 500] | 9 |
| Memory (MB) | [64, 80, 96, 112, 128, 160, 192, 256, 320, 512] | 10 |
| Timeout (ms) | [500, 800, 1000, 1500, 2000, 3000, 5000, 8000, 10000] | 9 |

Total discrete action space: 9 × 10 × 9 = **810 combinations** (previous experiments used 15 timeout bins = 1350 combinations; reduced to 9 after observing that bins below 500 ms are below Python startup overhead and bins above 10 s are never needed for competitive programming inputs).

A fixed overhead of 65 MB (guest OS + Python interpreter baseline) is added on top of the agent's chosen memory bin when computing allocation waste in the reward.

---

## Reward Function — `res_reward`

```
if exit_code == -1 and wall_time_ms == 0:          # VM boot failure
    res_reward = -4.0   (early return, no other terms)

else:
    base = +0.5   if tests_passed
         = -0.5   if tests_failed
         = -1.0   if timed_out
         = -3.0   if OOM-killed (exit_code == -9 or oom_killed)

    cpu_waste  = (allocated_cpu  - used_cpu_ms  / wall_ms) / allocated_cpu
    mem_waste  = (allocated_mem  - peak_mem_kb  / 1024)    / allocated_mem
    tms_waste  = (allocated_tms  - wall_ms)                / allocated_tms
    efficiency = -(cpu_waste × 0.5 + mem_waste × 2.0 + tms_waste × 0.2)
    latency    = -0.2 × max(0, wall_ms − 500) / 1000   # penalise slow execution

    res_reward = base + efficiency + latency
```

Memory waste is weighted 4× higher than CPU waste, reflecting that memory over-allocation is the dominant failure mode and is harder for the agent to correct from binary feedback alone.

| Scenario | Approximate res\_reward |
|---|---|
| Tight fit, fast, tests pass | ≈ +0.5 to +1.0 |
| Passes but 2 bins over-allocated | ≈ -1.0 to -1.5 |
| Times out | ≈ -2.0 to -3.0 |
| OOM killed | ≈ -3.5 |
| VM boot failure | -4.0 (exact) |

---

## Agent 2a — SAC Shared (Discrete Soft Actor-Critic, shared trunk)

### Architecture

```
State (9) ──► Linear(256) ──► ReLU ──► Linear(256) ──► ReLU ──► Linear(128) ──► ReLU
                                                                        │
                              ┌─────────────────────────────────────────┤
                              │                   │                     │
                        cpu_head              mem_head            timeout_head
                        Linear(128→9)      Linear(128→10)       Linear(128→15)
                        Softmax            Softmax               Softmax
```

All three action heads share a single trunk. The trunk learns a single 128-dimensional representation that all heads read from. This creates a representation bottleneck: the trunk must simultaneously encode signals useful for CPU allocation, memory allocation, and timeout selection. In practice, timeout and memory never develop strong independent signals because the shared representation is dominated by the cpu-relevant features (complexity, line count) which are seen most frequently.

| Property | Value |
|---|---|
| Trunk | 256 → 256 → 128 (ReLU activations) |
| Actor heads | cpu: 128→9, mem: 128→10, timeout: 128→15 |
| Critic | separate 256→256→128→1 (V-function) |
| Replay buffer | 20 000 transitions |
| Batch size | 256 |
| Warmup | 200 random episodes before first update |
| lr\_actor | 1e-4 (reduced from 3e-4 at ep 400) |
| lr\_critic | 2e-4 (reduced from 5e-4 at ep 400) |
| lr\_alpha | 1e-4 (reduced from 3e-4 at ep 400) |
| Target entropy | −0.70 × (log9 + log10 + log15) |
| Entropy annealing | After ep 500: log\_alpha × 0.995 every 10 ep, floor 0.01 |
| Soft target update | τ = 0.005 per gradient step |

### Training Results (394 problems → 1060 episodes)

| Episode block | Avg res\_reward | Notes |
|---|---|---|
| ep 1–100 (warmup) | −2.490 | Random exploration — baseline |
| ep 101–200 | −2.380 | First SAC updates |
| ep 201–300 | −2.307 | Gradual learning |
| ep 301–400 | −1.682 | Significant improvement |
| ep 401–500 | −1.973 | LR reduction + entropy target cut to 70% |
| ep 501–600 | −1.884 | Entropy annealing begins |
| ep 601–800 | −1.750 | Oscillation / policy churn |
| ep 801–860 | −1.580 | **Best window (ep 840–859: −1.580)** |
| ep 861–1060 | −1.700 | Plateau |

**Training summary:** 1060 episodes, overall avg res\_reward ≈ −2.038, positive rewards 117/1454 (8.0%), boot failures 283/1454 (19.5%).

**Policy churn at ep 560–599:** After peaking at ~−1.5 average, the policy regressed — timeout reverted to 3000 ms dominating, memory went back to 128 MB+ at 43% of picks. Diagnosis: the entropy term kept encouraging exploration even after the agent had found tight configs, pulling it away from good allocations. Interventions: (1) LR reduction at ep 400, (2) entropy annealing from ep 500, (3) resumed from ep 560 checkpoint rather than ep 600+ drift.

### Eval Results (60 unseen problems, `--use-ref`, deterministic)

| Category | n | Pass% | Avg res\_reward | Boot failures | Timeouts |
|---|---|---|---|---|---|
| easy | 9 | 66.7% | −0.716 | 2 | 0 |
| medium | 25 | 72.0% | −1.532 | 7 | 0 |
| hard | 9 | 66.7% | −0.346 | 0 | 0 |
| high\_memory | 17 | 41.2% | −3.082 | 10 | 0 |
| **OVERALL** | **60** | **61.7%** | **−1.671** | **19 (31.7%)** | **0** |

**Resource bin distribution (eval):**
```
CPU:     idx 8 (500mc) chosen 27/60 times — agent learned high CPU is safe
Memory:  idx 2 (96 MB) chosen 35/60 times — agents clusters at low-mid memory
Timeout: idx 5 (3000ms) chosen most often
```

---

## Agent 2b — SAC Factored (Discrete SAC, independent trunks)

### Motivation

The shared trunk in SAC-Shared creates a representation competition: all three heads must agree on what the 128-dimensional bottleneck encodes. Separate trunks allow each head to learn its own representation without interference. The hypothesis was that the memory head would develop sensitivity to memory-allocation-relevant code patterns independently.

### Architecture

```
                   ┌── cpu_trunk  ──► cpu_head  (→9)
State (9) ─────────┼── mem_trunk  ──► mem_head  (→10)
                   └── tms_trunk  ──► tms_head  (→15)

Each trunk: Linear(9→256) → ReLU → Dropout(0.1) → Linear(256→128) → ReLU
Each head:  Linear(128→N_actions)  →  Softmax
Critic:     Linear(9→256) → ReLU → Linear(256→128) → ReLU → Linear(128→1)
```

Three completely independent trunks, one per action dimension. Each trunk has its own parameters and gradients that are updated only by the loss signal from its corresponding head. The critic remains shared (single V-function).

| Property | Value |
|---|---|
| Each trunk | 256 → 128 (ReLU + Dropout 0.1) |
| Replay buffer | 20 000 transitions |
| Warmup | 200 episodes |
| lr\_actor (ep 1–600) | 1e-4 |
| lr\_critic (ep 1–600) | 2e-4 |
| lr\_actor (ep 601–1300) | 5e-5 (halved after churn observed) |
| lr\_critic (ep 601–1300) | 1e-4 |
| Other hyperparams | Identical to SAC-Shared |

### Training Results (394 problems → 1300 episodes)

| Episode block | Avg res\_reward | Boot failures | Notes |
|---|---|---|---|
| ep 1–100 (warmup) | −2.42 | 22% | Random baseline |
| ep 101–300 | −2.33 | 16% | Early learning |
| ep 301–600 | −2.09 | 19% | Improving with oscillation |
| ep 601–800 | −1.76 | 9% | **LR halved — boot failures halve** |
| ep 801–1000 | −1.79 | 7% | Stabilising |
| ep 1041–1080 | **−1.28** | **0%** | **Best 40-episode window** |
| ep 1201–1300 | −2.20 | 18% | Regression / churn |

**Training summary:** 1300 episodes, overall avg res\_reward ≈ −2.016, positive rewards 105/1300 (8.1%), boot failures 205/1300 (15.8%). Best 20-episode window: −1.28 at ep 1061–1080 with 0 boot failures.

**Policy churn at ep 1200+:** After the best window at ep 1061–1080, the policy collapsed at ep 1201–1220 (avg −2.88, 7/20 boot failures). Same mechanism as SAC-Shared: entropy kept driving exploration past convergence. The `sac_best.pt` checkpoint captures the pre-churn peak.

### Eval Results (120 episodes = 2 passes × 60 problems, `--use-ref`, deterministic)

| Category | n | Pass% | Avg res\_reward | Boot failures | Timeouts |
|---|---|---|---|---|---|
| easy | 27 | 96.3% | −1.298 | 1 | 0 |
| medium | 40 | 82.5% | −1.418 | 4 | 0 |
| hard | 21 | 66.7% | −1.556 | 1 | 0 |
| high\_memory | 32 | 43.8% | −2.727 | 18 | 0 |
| **OVERALL** | **120** | **72.5%** | **−1.764** | **24 (20.0%)** | **0** |

**Why factored is worse than shared on eval despite a better training peak:** The factored trunks allowed each head to specialise to the training distribution more aggressively. The memory head learned tight allocations that worked on the 394 training problems but under-allocated on unseen eval problems. SAC-Shared's shared trunk bottleneck acted as an implicit regulariser — the shared representation couldn't fully specialise to any one dimension, keeping allocations slightly conservative and more robust to distribution shift.

---

## Agent 2c — DQN (Deep Q-Network, three independent Q-networks)

### Architecture

```
State (9) ──► Q_cpu   : Linear(256) → ReLU → Dropout(0.1) → Linear(128) → ReLU → Linear(9)
State (9) ──► Q_mem   : Linear(256) → ReLU → Dropout(0.1) → Linear(128) → ReLU → Linear(10)
State (9) ──► Q_tms   : Linear(256) → ReLU → Dropout(0.1) → Linear(128) → ReLU → Linear(15)

Each Q-network has a corresponding target network (soft-updated τ=0.005).
Action = argmax over each Q-network independently.
```

DQN factorises the joint action space into three independent Q-functions: one for CPU, one for memory, one for timeout. This is a simplification — the true Q-value of the joint action `(cpu, mem, tms)` is not the sum of three independent Q-values — but it works well in practice because the three dimensions have mostly independent effect on the reward.

| Property | Value |
|---|---|
| Architecture per network | 256 → Dropout(0.1) → 128 → N\_actions |
| Parameter count | ~36 700 per Q-network, ~110 700 total |
| Target networks | Soft update τ=0.005 per gradient step |
| Replay buffer | 20 000 transitions |
| Batch size | 256 |
| Warmup | 200 random episodes |
| Exploration | ε-greedy: ε=1.0 → decays ×0.995 per ep after warmup, floor=0.05 |
| ε reaches minimum at | ≈ episode 1000 |
| Loss | Huber loss (smooth L1) on Bellman residual |
| lr | 2e-4 (all three networks, fixed) |
| γ | 0.99 |

**Key difference from SAC:** No actor network, no entropy regularisation term, no temperature parameter. DQN is purely value-based — it learns to estimate the expected cumulative reward for each action and selects greedily. Simpler signal, fewer moving parts, but no mechanism analogous to SAC's entropy term to prevent distribution overfitting.

### Training Results (394 problems → 1200 episodes, 200 warmup)

| Episode block | Avg res\_reward | Positive rewards | Boot failures |
|---|---|---|---|
| ep 1–200 (warmup) | −2.49 | 0% | ~20% |
| ep 201–400 | −2.15 | ~6% | 16% |
| ep 401–600 | −1.85 | 12% | 15% |
| ep 601–800 | −1.62 | 18% | 13% |
| ep 801–1000 | −1.48 | 20% | 12% |
| ep 1001–1040 (best) | **−0.75** | **35%** | **8%** |
| ep 1041–1200 | −1.55 | 15% | 14% |

**Training summary:** 1000 post-warmup episodes, overall avg res\_reward ≈ −1.499, positive rewards 167/1000 (16.7%), boot failures 143/1000 (14.3%). Best 20-episode window: −0.69 at ep 1021–1040.

**DQN wins training vs all SAC variants** — more than double the positive reward rate (16.7% vs ~8%), lower boot failure rate, and a better best window. This is because ε-greedy exploration is more systematic than SAC's entropy exploration in the early phases: DQN tries every action roughly equally before exploiting, whereas SAC's entropy term can keep revisiting sub-optimal actions.

### Eval Results (60 unseen problems, `--use-ref`, greedy argmax)

| Category | n | Pass% | Avg res\_reward | Boot failures | Timeouts |
|---|---|---|---|---|---|
| easy | 10 | 90.0% | −0.941 | 1 | 0 |
| medium | 21 | 76.2% | −1.449 | 4 | 0 |
| hard | 9 | 44.4% | −1.897 | 2 | 0 |
| high\_memory | 20 | 30.0% | −3.032 | 14 | 0 |
| **OVERALL** | **60** | **58.3%** | **−1.959** | **21 (35.0%)** | **0** |

**DQN is worst on eval despite best training:** Once ε hit its floor (~ep 1000), Q-values kept specialising to the 394 training problems with no entropy regulariser to prevent it. On unseen eval problems it defaulted to aggressive memory allocations that worked in training but failed on harder problems. The 44.4% hard pass rate (vs 66.7% for SAC-Shared) confirms overfitting to the training difficulty distribution.

---

## Agent 2d — PPO (Proximal Policy Optimization, factored actor)

### Architecture

```
Actor (factored trunks — identical to SAC-Factored actor):
  State (9) ──► cpu_trunk ──► cpu_head  (Softmax → 9)
  State (9) ──► mem_trunk ──► mem_head  (Softmax → 10)
  State (9) ──► tms_trunk ──► tms_head  (Softmax → 15)
  Each trunk: Linear(9→256) → ReLU → Dropout(0.1) → Linear(256→128) → ReLU

Critic (shared):
  State (9) ──► Linear(256) → ReLU → Linear(128) → ReLU → Linear(1)   [V-function]
```

PPO uses a rollout-based on-policy update rather than a replay buffer. The agent collects `rollout_steps=20` transitions, then runs `ppo_epochs=4` minibatch epochs over the collected rollout with a clipped surrogate objective.

**PPO clipped objective:**
```
ratio = π_θ(a|s) / π_θ_old(a|s)      (importance sampling ratio)
L_clip = E[min(ratio × A, clip(ratio, 1−ε, 1+ε) × A)]   ε = 0.20

actor_loss  = −L_clip
critic_loss = MSE(V(s), returns)
entropy     = −0.01 × H(π)   (small entropy bonus to prevent early collapse)
total_loss  = actor_loss + 0.5 × critic_loss − entropy_bonus
```

Advantages computed with Generalised Advantage Estimation (GAE, λ=0.95):
```
δ_t     = r_t + γ × V(s_{t+1}) × (1 − done_t) − V(s_t)
A_t_GAE = Σ_{k=0}^{T−t} (γλ)^k × δ_{t+k}
```
Advantages normalised to zero mean and unit variance before each PPO epoch. Gradient clipping at 0.5 for both actor and critic.

| Property | Value |
|---|---|
| Rollout steps | 20 (one update per 20 episodes) |
| PPO epochs per rollout | 4 |
| Clip epsilon | 0.20 |
| GAE lambda | 0.95 |
| Discount γ | 0.99 |
| Entropy coefficient | 0.01 |
| lr\_actor | 3e-4 |
| lr\_critic | 1e-3 |
| Gradient clip norm | 0.5 |
| Training pool | 836 problems (expanded from 394 for DQN/SAC) |

**Key difference from SAC/DQN:** PPO is **on-policy** — it updates exclusively on freshly collected data and discards it after the update. SAC and DQN are off-policy and reuse transitions from a replay buffer, which is more sample-efficient but introduces distribution shift between the collected data and the current policy. PPO avoids this via the clipped surrogate ratio but requires more environment interactions to collect fresh data.

### Training Results (836 problems → 753 episodes, 9-feature state)

| Episode block | Avg res\_reward | Pass rate | Boot failures |
|---|---|---|---|
| ep 1–100 | −2.393 | 87% | 13% |
| ep 101–200 | −2.207 | 87% | 13% |
| ep 201–300 | −1.873 | 87% | 8% |
| ep 301–400 | −1.637 | 84% | 13% |
| ep 401–500 | −1.530 | 88% | 13% |
| ep 501–600 | −0.737 | **89%** | 3% |
| ep 601–700 | −0.598 | 85% | **7%** |
| ep 681–700 (best 20) | **−0.336** | 90% | 5% |
| ep 701–753 | −1.097 | 92% | 1% |

**Training summary:** 753 episodes, overall avg res\_reward ≈ −1.615, positive rewards 142/753 (18.9%), boot failures 77/753 (10.2%).

PPO shows the fastest convergence trajectory of all four agents: by ep 500–600 it reaches average rewards in the −0.6 to −0.8 range — performance the SAC variants never sustained. This is attributable to the larger training pool (836 vs 394 problems): more diverse problems provide more generalised gradient signal per rollout. PPO also avoids the policy churn issue that plagued both SAC variants because the clipped surrogate explicitly prevents large policy updates.

### Eval Results (120 episodes = 2 passes × 60 problems, `--use-ref`, deterministic)

| Category | n | Pass% | Avg res\_reward | Boot failures | Timeouts |
|---|---|---|---|---|---|
| easy | 17 | 52.9% | −0.356 | 2 | 0 |
| medium | 43 | 67.4% | −0.936 | 13 | 0 |
| hard | 22 | 59.1% | −0.816 | 4 | 0 |
| high\_memory | 38 | 23.7% | −2.962 | 29 | 0 |
| **OVERALL** | **120** | **50.0%** | **−1.474** | **48 (40.0%)** | **0** |

**Note:** The PPO eval was run with the old 9-feature state vector (before the memory/timeout feature extensions). High\_memory boot failure rate of 29/38 (76%) reflects the absence of any memory-pattern signal in the state. The agent cannot distinguish a segment-tree problem from a simple loop problem.

---

## Agent 2d — PPO Phase 2: 18-Feature State + Mixed Pool Training

### What Changed

After the 9-feature PPO eval exposed the high-memory blind spot, two changes were made before retraining from scratch:

1. **18-feature state** — 9 memory-signal and timeout-signal features added (see Feature Engineering Progression below)
2. **Mixed training pool** — a curated 400-problem pool was assembled combining high-memory problems with easy/medium/hard problems to force the agent to encounter all difficulty tiers

### Pool Construction (400 problems)

```bash
# Built with /tmp/build_pool400.py
# 83 large-input problems (n ≥ 50 000 in test generator)  +  317 randomly sampled problems with generators
```

The 83 "large" problems were identified by scanning `cc_pool_cache.json` for generators containing `n = [50000+]` — graphs, trees, DP, DSU, segment trees. The remaining 317 were sampled from the full pool (easy/medium/hard) to maintain broad coverage.

### Training Run (ep 1–1420 on 400-problem mixed pool)

```bash
python3 -m online_rl.runner --agent ppo --use-ref --episodes 400 \
  --pool-cache online_rl/cc_pool_cache_400.json
# then resumed:
python3 -m online_rl.runner --agent ppo --use-ref --episodes 400 --resume \
  --pool-cache online_rl/cc_pool_cache_400.json
```

| Episode block | Avg res\_reward | Boot failures | High-reward (>0.5) |
|---|---|---|---|
| ep 1–100 | −0.162 | 11/80 (14%) | 21/80 (26%) |
| ep 101–200 | −0.317 | 17/100 (17%) | 30/100 (30%) |
| ep 201–300 | −0.074 | 13/100 (13%) | 45/100 (45%) |
| ep 301–400 | −0.387 | 22/100 (22%) | 49/100 (49%) |
| ep 401–420 (last 20) | **+0.479** | **1/20 (5%)** | **14/20 (70%)** |

The agent hit a clear inflection point around ep 400: boot failure rate collapsed from 22% → 5% and the high-reward rate (episodes ending with res\_reward > 0.5) reached 70%. Individual rewards in the last block were consistently +0.85–0.99.

### Eval Results at ep 1420 (113 episodes over 60-problem test set, `--use-ref`, deterministic)

```bash
python3 -m online_rl.runner --agent ppo --use-ref --eval \
  --checkpoint online_rl/checkpoints/ppo/res_ep_01420.pt \
  --pool-cache online_rl/cc_pool_cache_test.json
```

| Category | n | Pass% | Avg res\_reward | Boot failures |
|---|---|---|---|---|
| easy + medium + hard | 77 | 66% | −0.309 | 17/77 (22%) |
| **high\_memory** | **36** | **27%** | **−2.691** | **26/36 (72%)** |
| **OVERALL** | **113** | **54%** | **−1.068** | **43/113 (38%)** |

**Key finding — the high-memory blind spot persists:** the agent allocated 80 MB (mem\_idx=1, the lowest viable bin) for every single evaluation episode without exception. On easy/medium/hard problems this works well (peak ~12 MB → large efficiency bonus). On high-memory eval problems with large test cases (e.g. `cc_846_E` 5.7 MB stdin, `cc_963_B` 3.1 MB stdin, `cc_1208_D` 991 KB stdin), Python requires >80 MB to load and process the data, so the VM exits with code −1 (boot/OOM failure) before executing a single instruction.

The root cause is a **training distribution mismatch**: even the 83 "large-n" generators in the training pool only produced 12–20 MB peak memory usage in practice (Python lists for n=100 000 fit in ~20 MB). The agent never encountered a training episode where 80 MB actually failed, so it never learned to go higher.

### Phase 3 — Fresh PPO on Stratified Pool (in progress)

To force the agent to learn memory-tier discrimination, a new stratified pool of **301 problems** was assembled and PPO was restarted from scratch:

```bash
# Pool construction
# 81 large-input (n ≥ 50 000) + 60 easy (800–1200) + 80 medium (1300–1600) + 80 hard (1700+)
# Saved to cc_pool_cache_fresh.json

python3 -m online_rl.runner --agent ppo --use-ref --episodes 1000 \
  --pool-cache online_rl/cc_pool_cache_fresh.json
```

Pool breakdown:

| Bucket | Count | Rating range | Selection |
|---|---|---|---|
| Large-input | 81 | any | generators with `n ≥ 50 000` |
| Easy | 60 | 800–1200 | random from pool with generators |
| Medium | 80 | 1300–1600 | random from pool with generators |
| Hard | 80 | 1700+ | random from pool with generators |
| **Total** | **301** | — | — |

Results pending (training in progress).

---

## Synthetic High-Memory Problem Pool

### Motivation

All four agents failed on high-memory evaluation problems (pass rates 24–44%). The root cause: the training pool's CC generators produced only 12–20 MB peak usage even for n=100 000 inputs — the agent never saw a training episode where allocating ≥256 MB was actually necessary, so it never learned to do so.

To teach the agent that `has_collection_list=1` or `has_array_mult=1` means "allocate more memory", synthetic problems were created that **guaranteed OOM at low memory** and **guaranteed success at high memory**.

### Design Constraints

| Constraint | Requirement | How satisfied |
|---|---|---|
| Only fails from OOM | Must not timeout at low CPU | `bytearray(n_mb * 1024*1024)` is C-level, completes in <10 ms at any CPU bin |
| Triggers state features | `has_collection_list` or `has_array_mult` must be 1 | Tiny dummy Python pattern in code that fires the AST detector |
| Clean reward signal | No ambiguity between OOM and timeout | Bytearray is instant; only failure = OOM |
| Tiny stdin | No large pipe to break VM vsock | stdin is just `"n_mb\n"` (4 bytes) |

### Template Design

Two templates, both using `bytearray` for the actual allocation:

**ba_clist** — triggers `has_collection_list`:
```python
import sys
n_mb = int(sys.stdin.readline())
_sig = [[] for _ in range(min(n_mb, 1))]   # AST: ListComp with List elt → feature=1
data = bytearray(n_mb * 1024 * 1024)        # actual memory allocation, instant
print(len(data) % 1000000007)
```

**ba_amult** — triggers `has_array_mult`:
```python
import sys
n_mb = int(sys.stdin.readline())
_sig = [0] * min(n_mb, 1)                   # AST: BinOp(List, Mult) → feature=1
data = bytearray(n_mb * 1024 * 1024)
print(len(data) % 1000000007)
```

The dummy `_sig` lines execute in microseconds (at most 1 iteration). All memory cost comes from the `bytearray` call which is a single C `calloc`.

### OOM Thresholds (MEMORY_BINS = [64,80,96,112,128,160,192,256,320,512])

| n_mb | RSS on host | VM overhead | Total needed | OOM at | Success at |
|---|---|---|---|---|---|
| 128 | 137 MB | ~65 MB | ~193 MB | ≤192 MB (7 bins) | ≥256 MB (3 bins) |
| 150 | 159 MB | ~65 MB | ~215 MB | ≤192 MB (7 bins) | ≥256 MB (3 bins) |
| 192 | 201 MB | ~65 MB | ~257 MB | ≤256 MB (8 bins) | ≥320 MB (2 bins) |
| 224 | 233 MB | ~65 MB | ~289 MB | ≤256 MB (8 bins) | ≥320 MB (2 bins) |

### Pool Generation

```bash
python3 /tmp/gen_synthetic_highmem.py
# Output: online_rl/synthetic_highmem_pool.json
# 100 problems: 52 ba_clist + 48 ba_amult, RSS 137–234 MB, stdin 4 bytes each
```

### Reward Function Fix

The original OOM heuristic penalised high memory MORE than low memory:

```python
# OLD — perverse gradient: 512 MB → -4.0, 64 MB → -3.0
if wall == 0:
    if mem_mb <= 96:    return -3.0
    elif mem_mb <= 160: return -3.5
return -4.0
```

This trained the agent to reduce memory allocation on synthetic problems. Fixed to:

```python
# NEW — flat signal regardless of memory level
if wall == 0:
    return -3.0   # OOM signal at any memory level
return -4.0       # wall > 0 but exit=-1 → genuine infra failure
```

---

## Agent 2a Phase 2 — SAC Shared on 581-Problem Mixed Pool

### Architecture Changes vs Phase 1

| Property | Phase 1 | Phase 2 |
|---|---|---|
| State features | **9** | **18** |
| State dim | 9 | 18 |
| Timeout bins | 15 | **9** (reduced — bins below 500ms are below Python startup) |
| Architecture | State(9) → 256 → 256 → 128 → heads | State(18) → 256 → 256 → 128 → heads |
| Timeout head | Linear(128→15) | **Linear(128→9)** |
| Target entropy | −0.70 × (log9 + log10 + log15) | **−0.55 × (log9 + log10 + log9)** |
| Total action space | 9 × 10 × 15 = 1350 | **9 × 10 × 9 = 810** |
| Training pool | 394 CC problems | **581 (100 synthetic + 481 CC)** |
| SYN\_REPLAY\_FACTOR | — | **3 → 8** |

All other hyperparameters (lr, batch size, buffer size, warmup, τ) are identical to Phase 1.

### Motivation

After establishing that SAC-Shared was the best-generalising agent (−1.671 eval reward), a second training run was launched combining the synthetic high-memory pool with the original CC pool to teach both tight CC allocation and correct high-memory allocation simultaneously.

### Pool Construction (581 problems)

```bash
PYTHONPATH=/home/madiha/firecracker-rl python3 /tmp/build_mixed_pool.py
# Output: online_rl/cc_pool_cache_mixed581.json
```

| Bucket | Count | Selection |
|---|---|---|
| Synthetic high-memory | 100 | `bytearray` ba_clist + ba_amult problems (new) |
| Large-n CC | 81 | generators with `n ≥ 50 000` regex match |
| Easy CC | 100 | CF rating 800–1200 |
| Medium CC | 100 | CF rating 1300–1600 |
| Hard CC | 100 | CF rating 1700+ |
| High-timeout CC | 100 | `estimated_complexity ≥ 2` |
| **Total** | **581** | — |

### Key Changes vs Phase 1

| Change | Value | Reason |
|---|---|---|
| `MAX_RESOURCE_RETRIES` | 1 (was 3) | Retry loop was overwriting `--force-mem` override and creating ambiguous transitions |
| Reward OOM heuristic | flat −3.0 at wall=0 | Removed perverse gradient (see above) |
| `SYN_REPLAY_FACTOR` | 3 | Synthetic transitions pushed 3× to buffer so 100 synthetic episodes provide same gradient mass as 300 CC episodes |
| `--syn-factor N` flag | runtime override | Set to 1 to disable oversampling after synthetic learning stabilises |

### Training Run (complete)

```bash
# Initial force injection (8000ms timeout — later identified as biased)
python3 -m online_rl.runner --agent shared --use-ref \
  --pool-cache online_rl/synthetic_highmem_pool.json \
  --run-name sac_shared_581_bytearray \
  --episodes 30 --force-mem 320 --force-cpu 500 --force-timeout 8000

# Main training cycles (3× oversampling)
python3 -m online_rl.runner --agent shared --use-ref \
  --pool-cache online_rl/cc_pool_cache_mixed581.json \
  --run-name sac_shared_581_bytearray \
  --episodes 400 --resume

# Corrected force injection (2000ms — unbiased reward signal)
python3 -m online_rl.runner --agent shared --use-ref \
  --pool-cache online_rl/synthetic_highmem_pool.json \
  --run-name sac_shared_581_bytearray \
  --episodes 150 --force-mem 320 --force-cpu 200 --force-timeout 2000 \
  --syn-factor 8 --resume
```

Results saved to `online_rl/results/transitions_sac_shared_581_bytearray.jsonl`.

### Training Results (1481 total episodes: 944 CC + 537 synthetic)

**Synthetic allocation trend (50-episode windows):**

| Episode window | Avg Mem | ≥320MB | exit=0 | Avg Reward | Notes |
|---|---|---|---|---|---|
| ep 1–248 | 149 MB | 5/50 | 3/50 | −3.52 | Warmup + early training |
| ep 249–594 | 168 MB | 6/50 | 11/50 | −3.09 | Slow improvement |
| ep 598–723 | 254 MB | 33/50 | 32/50 | −2.40 | First 8000ms injection |
| ep 729–841 | 227 MB | 4/50 | 16/50 | −2.75 | Regression |
| ep 842–891 | 284 MB | 22/50 | 35/50 | −1.97 | Recovery |
| ep 892–986 | 281 MB | **39/50** | **39/50** | **−1.53** | Best window — 2000ms injections |
| ep 989–1215 | 256 MB | 25/50 | 23/50 | −2.00 | Sustained partial improvement |
| ep 1216–1481 | 269 MB | 23/37 | 18/37 | −1.61 | syn-factor 8 + final injection |

**CC performance (last 30 episodes):** avg reward −1.502 — consistently better than the fixed-128MB baseline.

### Force Injection — Key Finding

The initial injections used `--force-timeout 8000`. With bytearray executing in ~200–400 ms, this produced a large timeout over-allocation penalty:
- `r_tms = −0.2 × (8000ms bin − 400ms bin) ≈ −1.0 to −1.4`
- Q-function learned Q(s_syn, 320MB) ≈ −1.5 — including this penalty

This made 320MB appear artificially expensive. Fixed by switching to `--force-timeout 2000`:
- `r_tms ≈ −0.2 × 1 bin = −0.2`
- Q(s_syn, 320MB) ≈ −0.4 to −0.8 — correctly better than Q(s_syn, 160MB) ≈ −5.0

**Final 150-episode injection results (2000ms, syn-factor 8):** 116/150 exit=0, avg reward −0.429, some positive rewards (+0.787 max). 150 × 8 = 1200 buffer entries at Q(320MB) ≈ −0.43.

### Actor-Critic Disconnect (Core Limitation)

Despite 1200 buffer entries showing Q(s_syn, 320MB) ≈ −0.43, the greedy eval policy still chose 128–160 MB on all synthetic problems (0/14 success, avg mem 128 MB). Investigation revealed:

**Root cause:** In SAC, the critic (Q-function) and actor (policy network) are separate networks. Force injection updates the critic correctly. The actor is updated by sampling actions from the *current actor*, computing Q for those samples, and adjusting. If the actor has low probability for 320 MB (due to shared trunk being dominated by CC gradient), the 320 MB Q-signal has minimal weight in the actor update.

The actor update (discrete SAC, exact expectation):
```
L_π = Σ_a  π(a|s) × [α·log(π(a|s)) − Q(s,a)]
```
The term for 320 MB is weighted by π(320MB|s_syn). If the shared trunk outputs π(320MB) ≈ 0.05 (5% probability), the Q-advantage signal for 320 MB contributes very little gradient.

**Why the CC gradient wins:** The actor trunk is shared across all 18 states. CC episodes (~65% of training) consistently reward low memory. Synthetic episodes (~35%) push toward high memory. The CC gradient on the shared trunk's memory head consistently outweighs the synthetic gradient, even with 8× oversampling, because the CC episodes represent a stronger, more coherent training signal accumulated over hundreds of episodes.

### Entropy Reduction

Target entropy lowered from 0.70 → 0.55 × (log9 + log10 + log9) at episode 930:

```python
# config.py
"target_entropy": -(math.log(N_CPU) + math.log(N_MEMORY) + math.log(N_TIMEOUT)) * 0.55,
```

This sharpens the actor (less exploration, more exploitation of Q-values). Effect observed after ~50 episodes of adjustment. Did not resolve the synthetic memory allocation problem — the actor-critic disconnect persisted regardless of entropy level.

### Oversampling Implementation

```python
SYN_REPLAY_FACTOR = 3   # or 8 for aggressive phases

def _buf_push(buf, task_id: str, syn_factor: int, *args):
    k = syn_factor if task_id.startswith("synth_") else 1
    for _ in range(k):
        buf.push(*args)
```

Buffer capacity is 20 000. With 3× oversampling and ~17% synthetic episode rate, synthetic occupies ~37% of buffer. With 8× oversampling during injection phases, synthetic occupies ~58%.

---

## Baseline Comparison (80-problem CC eval, fixed sampling)

All agents evaluated on `cc_pool_cache_eval80.json` (80 problems, 20 per category, each problem seen exactly once, `--use-ref` mode):

| Agent | Features | Eval n | Avg Reward | Easy pass | Medium pass | Hard+HiMem pass | OOM/boot |
|---|---|---|---|---|---|---|---|
| **PPO-9feat** (836-prob, 730 ep) | 9 | 80† | **−1.386** | 18/20 (90%) | 12/20 (60%) | 21/40 (52%) | 23/80 (29%) |
| **SAC-Shared Phase 1** (394-prob, 1060 ep)‡ | 9 | 60 | −1.671 | 6/9 (67%) | 18/25 (72%) | 6/9 (67%) | 19/60 (32%) |
| **SAC-Factored** (394-prob, 1300 ep)‡ | 9 | 60 | −1.747 | 14/14 (100%) | 15/20 (75%) | 14/26 (54%) | 11/60 (18%) |
| **SAC-Shared Phase 2** (581-prob mixed, 1481 ep) | 18 | 81 | −1.835 | 16/20 (80%) | 13/20 (65%) | 22/41 (54%) | 25/81 (31%) |
| **Fixed baseline** (128MB / 125mc / 2000ms) | — | 80† | −1.998 | 19/20 (95%) | 12/20 (60%) | 21/40 (52%) | 22/80 (28%) |
| **DQN** (394-prob, 1200 ep)‡ | 9 | 60 | −1.959 | 9/10 (90%) | 16/21 (76%) | 10/29 (34%) | 21/60 (35%) |
| **PPO-18feat** (400-prob mixed, 1420 ep)‡ | 18 | 60 | −2.101 | 9/9 (100%) | 7/16 (44%) | 8/35 (23%) | 32/60 (53%) |
| **Min baseline** (64MB / 50mc / 500ms) | — | 80† | −3.000 | 0/20 (0%) | 0/20 (0%) | 0/40 (0%) | 80/80 (100%) |
| **Max baseline** (512MB / 500mc / 10000ms) | — | 80† | −3.637 | 19/20 (95%) | 18/20 (90%) | 35/40 (88%) | 3/80 (4%) |

† eval80 pool = `cc_pool_cache_eval80.json` (80 problems, 20 per category, fixed queue, each seen exactly once).  
‡ Random-sampled pool — per-category n varies, not directly comparable to eval80 results.

**Key findings:**

1. **PPO-9feat best reward (−1.386)** — tight 80MB memory + 3000ms timeout beats all agents despite 29% OOM rate; near-zero waste on successful runs compensates
2. **Max baseline (−3.637) is worse than Min baseline (−3.000)** — over-provisioning compounds 3-dimensional waste penalties (−2.1 mem + −1.6 tms + −1.0 cpu ≈ −4.7/ep) making it costlier than the flat −3.0 OOM floor
3. **SAC-Factored best pass rate (100% easy, 75% medium)** — separate trunks let the memory head specialise, but timeout over-allocation (38% at 8000ms) hurts reward
4. **All RL agents beat Fixed-128MB on reward** — even with similar pass rates (~63%), RL agents waste fewer resources on successful episodes
5. **PPO-18feat worst RL agent (−2.101, 53% boot fail)** — 18 features + smaller pool caused overfitting; collapsed to 100% 80MB / 100% 800ms allocations on eval
6. **High-memory blind spot persists** — no agent learned to allocate >128MB on graph/tree problems; the code-feature state vector lacks direct memory-need signals

**The synthetic training tradeoff:** Adding the synthetic pool improved CC performance on hard problems but did not achieve its primary goal (high-memory allocation). The shared trunk architecture cannot simultaneously learn "CC small programs → low memory" and "synthetic patterns → high memory" because these two signals conflict in the trunk gradient.

---

## Cross-Agent Comparison

> **Eval pool note:** SAC-Shared, SAC-Factored, DQN, and PPO-18feat were evaluated on their own randomly-sampled 60-problem pools (drawn from the CC training distribution, so per-category n varies). PPO-9feat and Fixed-128MB were evaluated on `cc_pool_cache_eval80.json` (80 problems, 20 per category, each seen exactly once). All runs used `--use-ref` (reference solution, no LLM) and greedy/deterministic policy.

### Training Performance

| Metric | SAC-Shared | SAC-Factored | DQN | PPO-9feat | PPO-18feat |
|---|---|---|---|---|---|
| Training pool | 394 CC | 394 CC | 394 CC | **836 CC** | 400 mixed |
| Total episodes | 1060 | 1300 | 1200 | 730 | 1420 |
| Avg res\_reward (training) | −2.038 | −2.016 | **−1.499** | −1.615 | −0.16→+0.48 |
| Boot failure rate (training) | 19.5% | 15.8% | 14.3% | 10.2% | **5%** |
| State features | 9 | 9 | 9 | 9 | **18** |
| Timeout bins | 15 | 15 | 15 | 15 | 9 |

---

### Overall Evaluation Results

† eval80 pool = `cc_pool_cache_eval80.json` (80 problems, 20 per category, fixed queue). Others = random-sampled 60-problem pools.

| Agent | Pool | n | Avg Reward | Completed | Boot Fails |
|---|---|---|---|---|---|
| **PPO-9feat** | eval80† | 80 | **−1.386** | 51 (64%) | 23 (29%) |
| **SAC-Shared Ph1** | 60-prob | 60 | −1.671 | 37 (62%) | 19 (32%) |
| **SAC-Factored** | 60-prob | 60 | −1.747 | 43 (72%) | 11 (18%) |
| **SAC-Shared Ph2** (18-feat, mixed pool) | 81-prob | 81 | −1.835 | 51 (63%) | 25 (31%) |
| **Fixed-128MB** (128MB/125mc/2000ms) | eval80† | 80 | −1.999 | 52 (65%) | 22 (28%) |
| **DQN** | 60-prob | 60 | −1.959 | 35 (58%) | 21 (35%) |
| **PPO-18feat** | 60-prob | 60 | −2.101 | 24 (40%) | 32 (53%) |
| **Min baseline** (64MB/50mc/500ms) | eval80† | 80 | −3.000 | **0 (0%)** | **80 (100%)** |
| **Max baseline** (512MB/500mc/10000ms) | eval80† | 80 | −3.637 | 72 (90%) | 3 (4%) |

---

### Per-Category Average Reward

| Category | SAC-Ph1 | SAC-Ph2 | SAC-Factored | DQN | PPO-18feat | PPO-9feat‡ | Fixed-128MB‡ | Min‡ | Max‡ |
|---|---|---|---|---|---|---|---|---|---|
| **easy** | −0.716 | −1.676 | −1.026 | −0.941 | **+0.293** | −0.939 | −1.567 | −3.000 | −3.964 |
| **medium** | −1.532 | −2.036 | −1.571 | −1.449 | −2.123 | −1.506 | −2.164 | −3.000 | −3.672 |
| **hard** | −2.135 | −1.816 | −2.271 | −2.680 | −2.706 | **−1.549** | −2.131 | −3.000 | −3.456 |
| **n easy** | 9 | 20 | 14 | 10 | 9 | 20 | 20 | 20 | 20 |
| **n medium** | 25 | 20 | 20 | 21 | 16 | 20 | 20 | 20 | 20 |
| **n hard** | 26 | 41 | 26 | 29 | 35 | 40 | 40 | 40 | 40 |

‡ eval80 pool only: hard and high\_memory counted together (both cf\_rating ≥ 1600).

---

### Per-Category Completion Rate

| Category | SAC-Ph1 | SAC-Ph2 | SAC-Factored | DQN | PPO-18feat | PPO-9feat‡ | Fixed-128MB‡ | Min‡ | Max‡ |
|---|---|---|---|---|---|---|---|---|---|
| **easy** | 6/9 (67%) | 16/20 (80%) | **14/14 (100%)** | 9/10 (90%) | **9/9 (100%)** | 18/20 (90%) | 19/20 (95%) | 0/20 (0%) | 19/20 (95%) |
| **medium** | 18/25 (72%) | 13/20 (65%) | 15/20 (75%) | 16/21 (76%) | 7/16 (44%) | 12/20 (60%) | 12/20 (60%) | 0/20 (0%) | 18/20 (90%) |
| **hard** | 13/26 (50%) | 22/41 (54%) | 14/26 (54%) | 10/29 (34%) | 8/35 (23%) | **21/40 (52%)** | 21/40 (52%) | 0/40 (0%) | 35/40 (88%) |

---

### Per-Category Boot Failures

| Category | SAC-Ph1 | SAC-Ph2 | SAC-Factored | DQN | PPO-18feat | PPO-9feat‡ | Fixed-128MB‡ | Min‡ | Max‡ |
|---|---|---|---|---|---|---|---|---|---|
| **easy** | 2/9 (22%) | 3/20 (15%) | **0/14 (0%)** | 1/10 (10%) | **0/9 (0%)** | 1/20 (5%) | **0/20 (0%)** | 20/20 (100%) | **0/20 (0%)** |
| **medium** | 7/25 (28%) | 7/20 (35%) | 3/20 (15%) | 4/21 (19%) | 9/16 (56%) | 7/20 (35%) | 7/20 (35%) | 20/20 (100%) | 1/20 (5%) |
| **hard** | 10/26 (38%) | 15/41 (37%) | 8/26 (31%) | 16/29 (55%) | **23/35 (66%)** | 15/40 (38%) | 15/40 (38%) | 40/40 (100%) | 2/40 (5%) |

---

### Resource Allocation Distribution

#### Memory (MB)

| Bin | SAC-Shared (n=60) | SAC-Factored (n=60) | DQN (n=60) | PPO-18feat (n=60) | PPO-9feat (n=80) | Fixed-128MB (n=80) |
|-----|-------------------|---------------------|------------|-------------------|------------------|-------------------|
| 64MB | 4 (7%) | 2 (3%) | 3 (5%) | — | — | — |
| **80MB** | 12 (20%) | 8 (13%) | 11 (18%) | **60 (100%)** | **76 (95%)** | 5 (6%) |
| 96MB | **35 (58%)** | **42 (70%)** | 26 (43%) | — | 4 (5%) | 1 (1%) |
| 112MB | 2 (3%) | 2 (3%) | 3 (5%) | — | — | 5 (6%) |
| **128MB** | 4 (7%) | — | 3 (5%) | — | — | **61 (76%)** |
| 160MB | — | 6 (10%) | 11 (18%) | — | — | 4 (5%) |
| 192MB | 1 (2%) | — | 1 (2%) | — | — | — |
| 256MB | 2 (3%) | — | — | — | — | 2 (2%) |
| 320MB | — | — | 1 (2%) | — | — | 2 (2%) |
| 512MB | — | — | 1 (2%) | — | — | — |

#### CPU (millicores)

| Bin | SAC-Shared (n=60) | SAC-Factored (n=60) | DQN (n=60) | PPO-18feat (n=60) | PPO-9feat (n=80) | Fixed-128MB (n=80) |
|-----|-------------------|---------------------|------------|-------------------|------------------|-------------------|
| 50mc | 2 (3%) | 2 (3%) | — | 2 (3%) | 4 (5%) | 3 (4%) |
| 75mc | 7 (12%) | — | 6 (10%) | 1 (2%) | **63 (79%)** | 2 (2%) |
| 100mc | 3 (5%) | 2 (3%) | 10 (17%) | — | 1 (1%) | 1 (1%) |
| 125mc | 3 (5%) | 2 (3%) | 10 (17%) | 1 (2%) | — | **64 (80%)** |
| 150mc | 7 (12%) | 7 (12%) | 2 (3%) | 3 (5%) | 2 (2%) | 1 (1%) |
| 175mc | 4 (7%) | 11 (18%) | — | **26 (43%)** | 1 (1%) | 1 (1%) |
| 200mc | 6 (10%) | 1 (2%) | 2 (3%) | — | — | 2 (2%) |
| 300mc | 1 (2%) | 3 (5%) | 2 (3%) | — | — | 3 (4%) |
| 500mc | **27 (45%)** | **32 (53%)** | **28 (47%)** | **27 (45%)** | 9 (11%) | 3 (4%) |

#### Timeout (ms)

| Bin | SAC-Shared (n=60) | SAC-Factored (n=60) | DQN (n=60) | PPO-18feat (n=60) | PPO-9feat (n=80) | Fixed-128MB (n=80) |
|-----|-------------------|---------------------|------------|-------------------|------------------|-------------------|
| 200ms‡ | **24 (40%)** | 1 (2%) | — | — | — | — |
| 300ms‡ | 1 (2%) | 15 (25%) | 19 (32%) | — | — | — |
| 400ms‡ | 1 (2%) | — | — | — | — | — |
| 500ms | — | — | — | — | — | 1 (1%) |
| 600ms‡ | 2 (3%) | 1 (2%) | 2 (3%) | — | — | — |
| 800ms | 1 (2%) | 1 (2%) | 1 (2%) | **60 (100%)** | 2 (2%) | 2 (2%) |
| 1000ms | 2 (3%) | — | 10 (17%) | — | 1 (1%) | 1 (1%) |
| 1500ms | 2 (3%) | 4 (7%) | 15 (25%) | — | — | 4 (5%) |
| 2000ms | 1 (2%) | — | 1 (2%) | — | — | **64 (80%)** |
| 3000ms | 6 (10%) | 6 (10%) | — | — | **70 (88%)** | 3 (4%) |
| 5000ms | 1 (2%) | 1 (2%) | 1 (2%) | — | 1 (1%) | 2 (2%) |
| 8000ms | 10 (17%) | **23 (38%)** | 6 (10%) | — | 5 (6%) | 3 (4%) |
| 10000ms | 5 (8%) | 2 (3%) | — | — | 1 (1%) | — |
| 15000ms | 2 (3%) | — | 2 (3%) | — | — | — |
| 20000ms | 1 (2%) | — | 2 (3%) | — | — | — |
| 30000ms | 1 (2%) | 6 (10%) | 1 (2%) | — | — | — |

‡ 200ms / 300ms / 400ms / 600ms were valid bins in the older 15-bin timeout config (used for SAC-Shared, SAC-Factored, DQN). Current 9-bin config starts at 500ms.

**Distribution highlights:**
- **SAC-Shared:** 58% at 96MB (right-sized), but timeout is scattered — 40% at 200ms to 17% at 8000ms, showing unresolved uncertainty
- **SAC-Factored:** 70% at 96MB (tight memory) but 38% at 8000ms timeout — memory learned, timeout did not
- **DQN:** widest spread across all bins — largest exploration variance, policy not converged
- **PPO-18feat:** fully collapsed — 100% at 80MB and 100% at 800ms, no diversity, zero learning of adaptation
- **PPO-9feat:** 95% at 80MB (very tight memory, causes 29% OOM) and 88% at 3000ms (timeout over-allocated due to bin config mismatch — trained on 15-bin config but evaluated with 9-bin, shifting action indices to higher ms values)
- **Fixed-128MB:** as expected, 76% at 128MB / 80% at 125mc / 80% at 2000ms — rigid but predictable

---

## Agent Architectures and Training

### SAC-Shared (Phase 1)

**Architecture:**
```
State (9) → [Shared Trunk] → 3 Heads
                 │
    Linear(9→256) → ReLU → Dropout(0.1)
    Linear(256→256) → ReLU → Dropout(0.1)
    Linear(256→128) → ReLU
                 │
    ┌────────────┼──────────────┐
 cpu_head    mem_head      tms_head
(128→9)     (128→10)      (128→15)
```
- Actor: shared trunk + 3 softmax heads (stochastic sampling during training, argmax at eval)
- Critic ×2: same shared trunk structure → 3 Q-value heads (per-action Q-values)
- Target critics ×2: soft-updated copies (τ=0.005)
- Learnable entropy temperature α (auto-tuned)

**Hyperparameters:**

| Param | Value |
|---|---|
| State dim | 9 |
| Action space | 9 × 10 × 15 = **1350** |
| Timeout bins | 15 (200ms → 30000ms) |
| lr\_actor | 1e-4 |
| lr\_critic | 2e-4 |
| lr\_alpha | 1e-4 |
| γ (gamma) | 0.99 |
| τ (soft update) | 0.005 |
| Batch size | 256 |
| Replay buffer | 20,000 |
| Warmup | 200 episodes |
| Target entropy coeff | 0.70 |
| Dropout | 0.1 (after layers 1 and 2) |
| Grad clip | 1.0 |

**Training trajectory (394 CC problems, 1060 episodes):**

| Episode range | Avg res\_reward (10-ep window) |
|---|---|
| 1–200 (warmup) | −2.7 → −2.4 |
| 200–400 | −2.1 → −1.8 |
| 400–600 | −1.6 → −1.5 |
| 600–800 | −1.5 (plateau) |
| 800–1060 | −1.5 → −1.55 (plateau, no further gain) |

Best checkpoint: `online_rl/checkpoints/shared/sac_best.pt`

---

### SAC-Shared (Phase 2)

Same architecture as Phase 1 but with expanded state and reduced action space. Key differences:

| Param | Phase 1 | Phase 2 |
|---|---|---|
| State dim | 9 | **18** |
| Action space | 1350 | **810** |
| Timeout bins | 15 (200ms→30s) | **9** (500ms→10s) |
| lr\_actor | 1e-4 | **5e-5** |
| lr\_critic | 2e-4 | **1e-4** |
| lr\_alpha | 1e-4 | **5e-5** |
| Target entropy coeff | 0.70 | **0.70 → 0.55** (changed at ep 930) |
| Training pool | 394 CC | **581 mixed** (481 CC + 100 synthetic) |
| Optimizer state saving | No | **Yes** |

Best checkpoint: `online_rl/checkpoints/sac_shared_581_bytearray/sac_best.pt`

---

### SAC-Factored

**Architecture:**
```
State (9) → 3 SEPARATE Trunks → 3 Heads

cpu_trunk:  Linear(9→256)→ReLU→Dropout(0.1)→Linear(256→128)→ReLU → cpu_head(128→9)
mem_trunk:  Linear(9→256)→ReLU→Dropout(0.1)→Linear(256→128)→ReLU → mem_head(128→10)
tms_trunk:  Linear(9→256)→ReLU→Dropout(0.1)→Linear(256→128)→ReLU → tms_head(128→15)
```
Each action dimension has its own trunk so it can develop an independent representation. Critic ×2 uses the same 3-trunk structure outputting per-action Q-values. Target critics ×2 soft-updated.

**Hyperparameters:** same as SAC-Shared Phase 1 (lr, γ, τ, batch, buffer, warmup) except:

| Param | Value |
|---|---|
| State dim | 9 |
| Action space | 9 × 10 × 15 = **1350** |
| Target entropy coeff | 0.70 |
| Trunk depth | 2 layers (256→128, shallower than shared) |

**Training trajectory (394 CC problems, 1300 episodes):**

| Episode range | Avg res\_reward |
|---|---|
| 1–200 (warmup) | ~−2.5 |
| 200–600 | −2.0 → −1.7 |
| 600–1000 | −1.6 → −1.5 (plateau) |
| 1000–1300 | −1.5 → −1.6 (mild policy churn) |

Best checkpoint: `online_rl/checkpoints/factored/sac_best.pt`

---

### DQN

**Architecture:**
```
State (9) → 3 SEPARATE Q-Networks

cpu_net:  Linear(9→256)→ReLU→Dropout(0.1)→Linear(256→128)→ReLU→Linear(128→9)
mem_net:  Linear(9→256)→ReLU→Dropout(0.1)→Linear(256→128)→ReLU→Linear(128→10)
tms_net:  Linear(9→256)→ReLU→Dropout(0.1)→Linear(256→128)→ReLU→Linear(128→15)
```
Each network outputs Q-values for all actions in its dimension. Target networks ×3 (same structure, soft-updated). No actor — action selection is ε-greedy argmax over Q-values.

**Hyperparameters:**

| Param | Value |
|---|---|
| State dim | 9 |
| Action space | 9 × 10 × 15 = **1350** |
| lr | 2e-4 |
| γ | 0.99 |
| τ (soft target update) | 0.005 |
| Batch size | 256 |
| Replay buffer | 20,000 |
| Warmup | 200 episodes |
| ε start | 1.0 |
| ε min | 0.05 |
| ε decay | 0.995 per episode |
| ε reaches min at | ~ep 600 |
| Dropout | 0.1 |
| Grad clip | 1.0 |

**Training trajectory (394 CC problems, 1200 episodes):**

| Episode range | Avg res\_reward |
|---|---|
| 1–200 (warmup) | ~−2.8 |
| 200–600 | −2.2 → −1.6 (ε decaying) |
| 600–900 | −1.5 → −1.4 (ε at floor, overfitting begins) |
| 900–1200 | −1.4 → −1.5 (specialising to training problems) |

Best checkpoint: `online_rl/checkpoints/dqn/sac_best.pt`

---

### PPO-9feat (836-problem pool)

**Architecture:**
```
State (9) → 3 SEPARATE Actor Trunks + 1 Shared Critic

Actor cpu_trunk:  Linear(9→256)→ReLU→Dropout(0.1)→Linear(256→128)→ReLU→cpu_head(128→9)
Actor mem_trunk:  Linear(9→256)→ReLU→Dropout(0.1)→Linear(256→128)→ReLU→mem_head(128→10)
Actor tms_trunk:  Linear(9→256)→ReLU→Dropout(0.1)→Linear(256→128)→ReLU→tms_head(128→15)

Critic (shared): Linear(9→256)→ReLU→Linear(256→128)→ReLU→Linear(128→1)  [no dropout]
```

**Hyperparameters:**

| Param | Value |
|---|---|
| State dim | 9 |
| Action space | 9 × 10 × 15 = **1350** |
| Timeout bins | 15 (200ms → 30000ms) |
| lr\_actor | 3e-4 |
| lr\_critic | 1e-3 |
| γ | 0.99 |
| GAE λ | 0.95 |
| Clip ε | 0.2 |
| PPO epochs per rollout | 4 |
| Rollout steps | 20 |
| Entropy coeff | 0.01 |
| Grad clip (actor) | 0.5 |
| Grad clip (critic) | 0.5 |
| No replay buffer — on-policy | — |

**Training trajectory (836 CC problems, 730 episodes):**

| Episode range | Avg res\_reward |
|---|---|
| 1–50 | −2.4 |
| 50–300 | −2.2 → −1.8 |
| 300–500 | −1.8 → −1.6 |
| 500–700 | −1.6 → −1.1 (strong improvement) |
| 700–730 | −1.0 (best checkpoint saved) |

Best checkpoint: `online_rl/checkpoints/ppo_836problems/sac_best.pt`

---

### PPO-18feat (mixed pool)

Same actor/critic architecture as PPO-9feat but with expanded state and current 9-bin timeout config:

| Param | PPO-9feat | PPO-18feat |
|---|---|---|
| State dim | 9 | **18** |
| Action space | 1350 | **810** |
| Timeout bins | 15 (200ms→30s) | **9** (500ms→10s) |
| Training pool | 836 CC | **400 mixed** (synthetic + CC) |
| Episodes | 730 | 1420 |

**Training trajectory (400 mixed problems, 1420+ episodes):**

| Episode range | Avg res\_reward |
|---|---|
| 1–300 | −2.4 → −1.8 |
| 300–700 | −1.5 → −0.8 |
| 700–1200 | −0.8 → +0.3 (strong improvement on training set) |
| 1200–1420 | +0.3 → −0.1 (policy churn, overfit to training) |

Note: the high training reward (+0.3 peak) reflects overfitting to the 400-problem mixed pool. Eval reward dropped to −2.10 on unseen problems, the worst of all agents.

Best checkpoint: `online_rl/checkpoints/ppo/sac_best.pt`

---

## Analysis

### Why Each Agent Generalises Differently

**SAC Shared — best overall eval reward (−1.671):**
SAC's entropy regularisation (`log_α` temperature) keeps the policy stochastic throughout training — even after convergence, the agent maintains a non-zero probability of non-greedy actions. This acts as an implicit regulariser: the policy cannot fully commit to the tightest allocations seen during training. Combined with the shared trunk bottleneck (which prevents any single dimension from over-specialising), SAC-Shared lands in the most conservative and generalisable part of the policy space.

**SAC Factored — best pass rate (72.5%) but mediocre reward (−1.764):**
Separate trunks succeeded in specialising each action head. The memory head learned to pick tighter memory bins on easy/medium problems (96.3% easy pass rate). However, this specialisation came at the cost of distribution generalisation: the memory head learned training-specific patterns and under-allocated on unseen hard and high\_memory problems. High pass rate combined with worse reward suggests the factored agent over-allocates on the problems it does pass (wasting memory), and fails completely on the rest.

**DQN — worst eval (−1.959) despite second-best training (−1.499):**
Without an entropy regulariser, once ε reaches its floor (~ep 1000), Q-values specialise directly to the training problem distribution. On unseen problems, the greedy policy defaults to allocations that were correct in training but are wrong for different problem structures. The 44.4% hard pass rate (vs 66.7% for SAC-Shared) and 14 high\_memory boot failures (vs 10 for SAC-Shared) confirm classic Q-value overfitting.

**PPO — numerically best eval reward (−1.474) with caveats:**
PPO's on-policy updates and clipped surrogate prevent the catastrophic policy churn seen in both SAC variants. The 40% boot failure rate on eval despite a 10% boot failure rate during training reveals the main weakness: with 9 features and no memory signals, the PPO policy learned to under-allocate memory on unseen high\_memory problems. The better avg reward (−1.474 vs −1.671) is partly attributable to the 836-problem training pool providing richer gradient signal, not just the algorithm.

### The High Memory Problem

Every agent fails on `high_memory` problems (graph/tree/DP) — pass rates of 41%, 44%, 30%, and 24% respectively. The root cause is the 9-feature state vector:

- **What the state captures:** algorithmic complexity (loop depth, cyclomatic complexity, recursion), code length, recent execution history
- **What it misses:** memory allocation patterns — segment trees (`[0]*(4*n)`), BIT arrays (`[0]*(n+1)`), adjacency lists (`[[] for _ in range(n)]`), adjacency sets (`[{i} for i in range(n)}`), `defaultdict`, `deque`, `heapq`

Without these signals, the agent cannot distinguish a 50-line graph BFS (needs 256 MB) from a 50-line string problem (needs 80 MB). All it sees is similar complexity scores and line counts. The 9-feature state is insufficient for memory allocation.

**Fix applied after all evaluations:** 9 memory-signal features and 4 timeout-signal features added to `state_builder.py`, expanding the state to 18 features. PPO is currently being retrained from scratch with 18 features.

### Boot Failure Root Cause

All boot failures have the same signature: `exit_code = −1`, `wall_time_ms = 0`, `mem_peak_kb = 0`. The Firecracker guest OS + Python runtime requires a baseline of ~65 MB regardless of the program. If the agent chooses memory bin idx 0 (64 MB) or idx 1 (80 MB), the guest cannot boot. Agents learn to avoid this eventually, but the learning signal is slow: the boot failure reward (−4.0) is the same regardless of how far below the boot threshold the agent went, providing no gradient direction for "how much more memory to add".

---

## Feature Engineering Progression

| Version | Features | State dim | Agents trained | Key change |
|---|---|---|---|---|
| v1 | 9 original (complexity + rolling) | 9 | SAC-Shared, SAC-Factored, DQN, PPO (old) | Initial design |
| v2 | +5 memory signals | 14 | — | `uses_defaultdict`, `uses_deque`, `uses_heapq`, `has_array_mult`, `has_collection_list` |
| v3 | +4 timeout signals | **18** | PPO (current retraining) | `uses_itertools`, `has_lru_cache`, `sort_call_count`, `has_while_true` |

**Memory signal detection logic (v2):**
- `uses_heapq` / `uses_deque` / `uses_defaultdict` — import detection via AST `Import` / `ImportFrom` nodes
- `has_array_mult` — detects `[x]*N` pattern (BinOp/Mult where left or right is a List literal) — covers segment trees `[0]*(4*n)`, BIT arrays `[0]*(n+1)`, DP tables `[inf]*(n+1)`
- `has_collection_list` — detects `[[] for _ in range(n)]` (ListComp where elt is List/Set/Dict) — covers adjacency lists and set graphs; also fires on `sys.setrecursionlimit` calls (indicator of deep recursive solutions)

**Timeout signal detection logic (v3):**
- `uses_itertools` — import detection (permutations/combinations → exponential time)
- `has_lru_cache` — detects `@lru_cache` / `@cache` decorator or `functools.lru_cache` attribute call
- `sort_call_count` — counts `sorted(...)` calls (Name node) and `.sort()` method calls (Attribute node); scaled by ÷10
- `has_while_true` — detects `while True:` loop (While node where test is Constant(True))

---

## Training Pool

| Pool | Size | Source | Used by |
|---|---|---|---|
| `cc_pool_cache.json` | 394 problems | deepmind/code\_contests | SAC-Shared, SAC-Factored, DQN |
| `cc_pool_cache.json` (expanded) | 836 problems | deepmind/code\_contests | PPO (9-feat, old) |
| `cc_pool_cache_400.json` | 400 problems | 83 large-n + 317 random | PPO (18-feat Phase 2, ep 1–1420) |
| `cc_pool_cache_fresh.json` | 301 problems | 81 large + 60 easy + 80 med + 80 hard | PPO (18-feat Phase 3, in progress) |
| `cc_pool_cache_test.json` | 60 problems | deepmind/code\_contests | Eval (all agents) |

**Pool filters:** Python 3 reference solution required, generated test inputs required, not already in training pool (for eval set).

---

## Evaluation Sets

### CC-only Eval Pool — `cc_pool_cache_eval80.json` (80 problems)

Used for the final baseline comparison. 80 problems, 20 per category, all CC (no synthetic). Each problem seen exactly once per eval run (fixed queue sampling, not random.choice).

| Category | n | CF rating | Purpose |
|---|---|---|---|
| easy | 20 | 800–1199 | Verify no over-allocation on simple problems |
| medium | 20 | 1200–1599 | Mid-range generalisation |
| hard | 20 | ≥1600 | Complex unseen problems |
| high\_memory | 20 | any | Stress-test memory allocation (graphs, trees, DP) |

### Full Eval Pool — `cc_pool_cache_test.json` (90 problems)

| Category | n | CF rating | Purpose |
|---|---|---|---|
| easy | 20 | 800–1199 | Simple problems |
| medium | 20 | 1200–1599 | Mid-range |
| hard | 20 | ≥1600 | Complex problems |
| high\_memory | 20 | any | Memory stress-test |
| synthetic\_highmem | 10 | — | Unseen bytearray problems (seeds 14–15, not in training pool) |

The synthetic eval problems use the same `ba_clist` and `ba_amult` templates as the training pool but with different seeds (14, 15) — never seen during training. They serve as a direct test of whether the agent learned to allocate high memory when `has_collection_list=1` or `has_array_mult=1`.

Each CC problem has a hand-crafted stress-test generator stored in the JSON. All 80 CC generators verified against reference solutions.

---

## Eval Protocol

```bash
# Agent 2 only (no LLM calls, reference solution used directly)
python3 -m online_rl.runner \
  --agent {shared|factored|dqn|ppo} \
  --checkpoint online_rl/checkpoints/{agent}/sac_best.pt \
  --eval \
  --use-ref \
  --pool-cache online_rl/cc_pool_cache_test.json \
  --episodes 60
```

- `--eval`: sets deterministic action selection (argmax, no sampling), disables weight updates, disables checkpoint saving, logs to `eval_transitions_{agent}.jsonl`
- `--use-ref`: uses `ref_solution` from pool cache directly as code — bypasses Agent 1 (bandit) and PREP VM entirely, zero API cost
- `--pool-cache`: loads the 60-problem eval set instead of training pool

---

## Running

### Train from scratch

```bash
# SAC Shared
python3 -m online_rl.runner --agent shared --use-ref --episodes 1000

# SAC Factored
python3 -m online_rl.runner --agent factored --use-ref --episodes 1000

# DQN
python3 -m online_rl.runner --agent dqn --use-ref --episodes 1000

# PPO (18-feature, current)
python3 -m online_rl.runner --agent ppo --use-ref --episodes 1000
```

### Resume training

```bash
python3 -m online_rl.runner --agent ppo --use-ref --episodes 500 --resume
```

### Evaluate

```bash
python3 -m online_rl.runner \
  --agent ppo \
  --checkpoint online_rl/checkpoints/ppo/sac_best.pt \
  --eval --use-ref \
  --pool-cache online_rl/cc_pool_cache_test.json \
  --episodes 60
```

### Print eval results

```bash
python3 online_rl/eval_results.py
```

---

## Runner Flags

| Flag | Default | Description |
|---|---|---|
| `--agent` | shared | Agent type: `shared`, `factored`, `dqn`, `ppo` |
| `--episodes N` | 3000 | Number of episodes |
| `--resume` | off | Resume training from latest checkpoint in `checkpoints/{agent}/` |
| `--checkpoint PATH` | off | Load specific `.pt` file (training or eval) |
| `--eval` | off | Deterministic actions, no updates, log to `eval_transitions_{agent}.jsonl` |
| `--use-ref` | off | Use `ref_solution` as code — skip bandit + PREP VM (zero API cost) |
| `--pool-cache PATH` | cc\_pool\_cache.json | Alternate problem pool |
| `--no-code-cache` | off | Skip code cache — always call LLM fresh |
| `--pool N` | all | Problems to use from pool cache |
| `--dry-run` | off | Synthetic execution — no LLM, no VMs |
| `--force-cpu N` | off | Override agent CPU decision |
| `--force-mem N` | off | Override agent memory decision |
| `--force-timeout N` | off | Override agent timeout decision |
| `--task ID` | off | Run only one problem (enables verbose) |
| `--run-name NAME` | off | Save transitions to `results/transitions_{NAME}.jsonl`, checkpoints to `checkpoints/{NAME}/` |
| `--syn-factor N` | 3 | Times to push each synthetic transition to replay buffer (1 = disable oversampling) |

---

## File Structure

```
firecracker-rl/
│
├── online_rl/
│   ├── runner.py                        # Main training/eval loop
│   ├── config.py                        # Hyperparameters, bins, feature cols (18-dim now)
│   ├── state_builder.py                 # AST feature extraction + state vector builder (18 features)
│   ├── sac_agent.py                     # Discrete SAC — shared trunk variant
│   ├── ppo_agent.py                     # PPO — factored actor + shared critic
│   ├── bandit.py                        # Thompson sampling bandit (Agent 1)
│   ├── rewards.py                       # llm_reward + res_reward functions
│   ├── llm_caller.py                    # LLM routing (free/sonnet/opus)
│   ├── code_cache.py                    # Per-task LLM code cache
│   ├── problem_loader.py                # Loads pool cache JSON
│   ├── prepare_inputs.py                # Add problems from HuggingFace to pool
│   │
│   ├── fetch_training_problems.py       # Expanded training pool to 836 problems
│   ├── fetch_eval_problems.py           # Fetches 60 unseen eval problems
│   ├── generate_stress_tests_training.py  # Stress generators for training problems
│   ├── generate_stress_tests_eval.py    # Stress generators for eval problems (60/60)
│   ├── eval_results.py                  # Print result tables from eval_transitions.jsonl
│   │
│   ├── cc_pool_cache.json               # 836-problem training pool
│   ├── cc_pool_cache_test.json          # 60-problem eval set with embedded generators
│   │
│   ├── checkpoints/
│   │   ├── ppo/                         # PPO checkpoints (current 18-feature retraining)
│   │   ├── ppo_836problems/             # PPO checkpoints (old 9-feature run, 836 probs)
│   │   ├── ppo_400problems/             # PPO checkpoints (old 9-feature run, 394 probs)
│   │   ├── factored/                    # SAC-Factored checkpoints
│   │   └── dqn/                         # DQN checkpoints
│   │
│   └── results/
│       ├── transitions.jsonl            # SAC-Shared training transitions
│       ├── transitions_ppo.jsonl        # PPO training transitions (753 ep, 9-feat)
│       ├── curve.jsonl                  # Per-episode reward curve
│       ├── eval_transitions.jsonl       # SAC-Shared eval (60 ep)
│       ├── eval_transitions_factored.jsonl  # SAC-Factored eval (120 ep)
│       ├── eval_transitions_dqn.jsonl   # DQN eval (60 ep)
│       └── eval_transitions_ppo.jsonl   # PPO eval (120 ep, 9-feat state)
│
├── vmlinux                              # Guest kernel image
├── rootfs.ext4                          # Guest root filesystem
└── firecracker                          # Firecracker binary
```

---

## Changelog

### 2026-06-22 — SAC Shared Phase 2 complete + baseline comparison

**Complete training run:** 1481 episodes (944 CC + 537 synthetic) on `cc_pool_cache_mixed581.json`. Multiple force injection cycles with `--force-mem 320`. Key finding: initial injections with `--force-timeout 8000` biased Q(320MB) negatively due to timeout over-allocation penalty; corrected to `--force-timeout 2000` which gave avg reward −0.429 on synthetic (vs −1.5 before).

**Actor-critic disconnect confirmed:** Despite 1200 buffer entries (150 episodes × syn-factor 8) at Q(320MB) ≈ −0.43, greedy eval policy chose 128 MB on 14/14 synthetic eval problems (0 success). Root cause: SAC actor update is weighted by current actor probability — since CC gradient keeps π(320MB|s_syn) near 5%, the Q-advantage signal for 320MB barely reaches actor gradient.

**Entropy reduction:** `target_entropy` changed from 0.70 → 0.55 × max entropy at episode 930. Did not resolve actor-critic disconnect.

**`--syn-factor 8` flag:** raised from 3 to 8 for final injection phase giving 1200 synthetic buffer entries. Added as CLI flag `--syn-factor N`.

**Eval sampling fixed:** `runner.py` now uses a shuffled queue in eval mode (`--eval`) so each problem is seen exactly once per cycle instead of random.choice with replacement. Old eval results had uneven category distribution (e.g. 9 easy, 25 medium for a 60-episode eval over 20+20+20 problems).

**Baseline comparison:** Fixed-resource baseline (128MB / 125mc / 2000ms) established on 80-problem CC eval pool. SAC Phase 2 beats it overall (−1.835 vs −1.998) and on hard problems (76% vs 70% pass rate), confirming the RL agent learned meaningful resource allocation despite the high-memory limitation.

**New eval pool:** `cc_pool_cache_eval80.json` (80 CC-only problems, 20 per category) + `cc_pool_cache_test.json` expanded to 90 problems by adding 10 synthetic_highmem eval problems (seeds 14–15, unseen) and 10 extra easy + 10 extra hard CC problems.

---

### 2026-06-22 — Synthetic high-memory pool + SAC Shared Phase 2 initial

**`/tmp/gen_synthetic_highmem.py`** — redesigned synthetic problem generator. Replaced pure-Python `adjlist` and `dp2d` templates (which timed out at low CPU and produced ambiguous exit=-1 signals) with `bytearray`-based templates that are instant at any CPU and only fail from OOM. Two templates: `ba_clist` (triggers `has_collection_list`) and `ba_amult` (triggers `has_array_mult`). Sizes: n_mb ∈ {128,150,192,224} → RSS 137–234 MB. Regenerated `online_rl/synthetic_highmem_pool.json` (100 problems).

**`online_rl/rewards.py`** — removed perverse OOM gradient. Old heuristic gave exit=-1+wall=0 a worse reward at higher memory (512 MB → −4.0, 64 MB → −3.0), which trained the agent to reduce memory on synthetic problems. Fixed to always return −3.0 for wall=0 failures regardless of memory level.

**`online_rl/cc_pool_cache_mixed581.json`** (new) — 581-problem mixed pool: 100 synthetic + 81 large-n CC + 100 easy + 100 medium + 100 hard + 100 high-timeout.

**`online_rl/runner.py`**:
- `SYN_REPLAY_FACTOR = 3` — push synthetic transitions 3× to buffer to counter CC volume dominance
- `_buf_push()` helper wrapping all three `res_buffer.push` sites
- `--run-name` flag — named run saves to `results/transitions_{name}.jsonl` and `checkpoints/{name}/`
- `--syn-factor N` flag — runtime override for `SYN_REPLAY_FACTOR` (pass 1 to disable after synthetic stabilises)
- `MAX_RESOURCE_RETRIES = 1` (was 3) — prevents retry loop from overwriting forced resource actions

**SAC Shared Phase 2 training** (`sac_shared_581_bytearray`): 582 episodes on mixed pool. CC last 20 training avg: −1.39 (beats Phase 1 best of −1.55). Synthetic success stalled at ~15% due to CC volume dominance — addressed with 3× oversampling + force injection of 30 synthetic successes.

---

### 2026-06-21 — Phase 3: stratified 301-problem pool + fresh PPO restart

**`online_rl/cc_pool_cache_fresh.json`** (new) — 301-problem stratified pool: 81 large-input (n ≥ 50 000 in generator), 60 easy (CF 800–1200), 80 medium (CF 1300–1600), 80 hard (CF 1700+). Assembled by scanning `cc_pool_cache.json` for generator size and rating tier. PPO restarted from scratch on this pool.

**`online_rl/cc_pool_cache_400.json`** (archived) — 400-problem mixed pool used for Phase 2 training (ep 1–1420). 83 large-n problems + 317 randomly sampled problems with generators.

**Phase 2 eval (ep 1420 checkpoint):** avg res\_reward −1.068, boot failures 38% overall, 72% on high\_memory — same blind spot as 9-feature agents. Agent converged to always allocating 80 MB because training generators never produced >20 MB peak usage even for n=100 000 inputs.

---

### 2026-06-21 — State expansion to 18 features + timeout bin reduction

**`online_rl/state_builder.py`**
- Added 5 memory-signal features: `uses_defaultdict`, `uses_deque`, `uses_heapq`, `has_array_mult`, `has_collection_list`
- Added 4 timeout-signal features: `uses_itertools`, `has_lru_cache`, `sort_call_count`, `has_while_true`
- State dimension: 9 → 18; `_RAW_SCALES` extended to 18 entries

**`online_rl/config.py`**
- `SAC_FEATURE_COLS` expanded from 9 → 18 entries
- `state_dim` updated to 18 in `RES_SAC_CONFIG`, `DQN_CONFIG`, `PPO_CONFIG`
- `TIMEOUT_BINS` reduced from 15 bins to 9 bins: `[500, 800, 1000, 1500, 2000, 3000, 5000, 8000, 10000]`
- `N_TIMEOUT` updated to 9; `target_entropy` auto-updates from variable

Old PPO checkpoints (9-feature) moved to `checkpoints/ppo_836problems/`. Old scaler deleted. PPO retraining started from scratch with 18-feature state.

---

### 2026-06-16 — Eval pipeline + 60-problem test set

**`online_rl/fetch_eval_problems.py`** (new) — fetches 60 unseen eval problems into `cc_pool_cache_test.json`

**`online_rl/generate_stress_tests_eval.py`** (new) — 60 hand-crafted stress generators, all 60/60 verified and embedded

**`online_rl/runner.py`** — `--pool-cache`, `--no-code-cache`, `--use-ref` flags added

**`online_rl/eval_results.py`** (new) — prints RL1 + RL2 result tables from eval transitions

---

### 2026-06-10 — Code caching + pool expansion

**`online_rl/code_cache.py`** (new) — persistent JSON cache per task\_id; LLM calls reduced from ~10 000 to ~836

**`online_rl/prepare_inputs.py`** — `--add-new N` flag: finds N new problems from HuggingFace not in pool

---

### 2026-06-01 — Entropy annealing + optimizer checkpointing

**`online_rl/config.py`** — target entropy to 70% of maximum, LR reductions at ep 400

**`online_rl/sac_agent.py`** — saves/loads optimizer states; `load()` accepts config override

**`online_rl/runner.py`** — `--checkpoint` flag, entropy annealing (ep 500+), best-model tracking (`sac_best.pt`)

---

### 2026-05-27 — Initial online RL system

Two-agent online RL: Thompson sampling bandit (Agent 1) + Discrete SAC (Agent 2). Replaced offline data collection pipeline from May 2026.
