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

## State Space — 9 Features

All four Agent 2 variants use the same 9-dimensional state vector. Features are split into two groups: static code features (extracted once per problem via AST analysis) and rolling execution history (updated after each execution via exponential moving average).

| # | Feature | Type | Scale | Description |
|---|---|---|---|---|
| 0 | `cyclomatic_complexity` | Static | ÷50 | 1 + number of branches, loops, comprehensions in AST |
| 1 | `max_loop_depth` | Static | ÷10 | Maximum nesting depth of `for`/`while` loops |
| 2 | `estimated_complexity` | Static | ÷5 | Ordinal 0–3: O(1)→O(n)→O(n²)→O(n³+) derived from CC and loop depth |
| 3 | `has_recursion` | Static | ÷1 | Binary: any function name appears in its own call graph |
| 4 | `ast_node_count` | Static | ÷1000 | Total AST node count — proxy for code size |
| 5 | `line_count` | Static | ÷200 | Non-blank line count |
| 6 | `recent_success_rate` | Rolling | ÷1 | EMA(α=0.1) of pass/fail — recent execution success rate |
| 7 | `recent_mean_cpu_used` | Rolling | ÷200 | EMA of `cpu_user_ms` from CONFIG VM |
| 8 | `recent_mean_mem_used` | Rolling | ÷50000 | EMA of `mem_peak_kb` from CONFIG VM |

Rolling features start at zero and update via:
```
new_val = 0.1 × current_measurement + 0.9 × old_ema
```

The scaler is fitted online after 200 warmup episodes using `sklearn.StandardScaler`, then applied for the remainder of training.

**Limitation of the 9-feature set:** These features capture algorithmic complexity proxies (loop depth, recursion, cyclomatic complexity) but contain no explicit signals for memory-intensive patterns (segment trees, adjacency lists, BIT arrays) or timeout-risky patterns (itertools permutations, unbounded while loops). This blind spot is the primary cause of high\_memory evaluation failures across all four agents.

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

## Cross-Agent Comparison

### Training Performance

| Metric | SAC Shared | SAC Factored | DQN | PPO (9-feat) | PPO (18-feat Phase 2) |
|---|---|---|---|---|---|
| Training pool size | 394 | 394 | 394 | **836** | 400 mixed |
| Total episodes | 1060 | 1300 | 1200 (200 warmup) | 753 | 1420 |
| Overall avg res\_reward | −2.038 | −2.016 | **−1.499** | −1.615 | −0.162→+0.479 |
| Boot failure rate | 19.5% | 15.8% | 14.3% | 10.2% | **5% (last 20 ep)** |
| State features | 9 | 9 | 9 | 9 | **18** |

### Evaluation Performance (60 unseen problems, `--use-ref`)

| Metric | SAC Shared | SAC Factored | DQN | PPO (9-feat) | PPO (18-feat Phase 2) |
|---|---|---|---|---|---|
| Avg res\_reward | **−1.671** | −1.764 | −1.959 | −1.474\* | −1.068 |
| Pass rate | **61.7%** | 72.5%† | 58.3% | 50.0%† | 54% |
| Boot failures | 19/60 (31.7%) | **12/60 (20.0%)**† | 21/60 (35.0%) | 48/120 (40.0%)† | 43/113 (38%) |
| high\_memory avg\_reward | −3.082 | −2.727 | −3.032 | −2.962 | **−2.691** |
| high\_memory boot fails | — | — | — | 29/38 (76%) | 26/36 (72%) |

\* PPO (9-feat) eval avg best numerically but trained on more problems (836 vs 394)  
† SAC Factored and PPO (9-feat) ran 120 episodes (2 passes), others ran 60

### Per-Category Evaluation

| Category | SAC Shared | SAC Factored | DQN | PPO |
|---|---|---|---|---|
| **easy** pass% | 66.7% | **96.3%** | 90.0% | 52.9% |
| **easy** avg\_reward | −0.716 | −1.298 | −0.941 | **−0.356** |
| **medium** pass% | **72.0%** | 82.5% | 76.2% | 67.4% |
| **medium** avg\_reward | −1.532 | −1.418 | −1.449 | **−0.936** |
| **hard** pass% | **66.7%** | 66.7% | 44.4% | 59.1% |
| **hard** avg\_reward | **−0.346** | −1.556 | −1.897 | −0.816 |
| **high\_memory** pass% | **41.2%** | 43.8% | 30.0% | 23.7% |
| **high\_memory** avg\_reward | **−3.082** | −2.727 | −3.032 | −2.962 |

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

## Evaluation Set

60 problems never seen during training, split into four categories:

| Category | n | CF rating | CF tags | Purpose |
|---|---|---|---|---|
| high\_memory | 20 | any | graphs, trees, dfs, dp, data structures | Stress-test memory allocation |
| easy | 10 | 800–1199 | any | Verify agent doesn't over-allocate on simple problems |
| medium | 20 | 1200–1599 | any | Mid-range generalisation |
| hard | 10 | ≥1600 | any | Performance on complex unseen problems |

Each problem has a hand-crafted stress-test generator (stored as `test_case_generator` in `cc_pool_cache_test.json`) that produces large inputs to stress the reference solution. All 60/60 generators were verified against their reference solutions before embedding.

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
