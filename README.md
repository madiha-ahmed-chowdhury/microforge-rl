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
  Dataset (MBPP /  ───► │  1. Load coding prompt        │
  HumanEval)            │                               │
                        │  2. Call MiniMax LLM API ────►│──► MiniMax API (external)
                        │     ◄── generated Python code │
                        │                               │
                        │  3. For each resource config: │
                        │     ┌─────────────────────┐   │
                        │     │  Fresh Firecracker   │   │
                        │     │  microVM             │   │
                        │     │  (250mc / 128MB)     │   │
                        │     │                      │   │
                        │     │  execute code        │   │
                        │     │  ← metrics           │   │
                        │     └─────────────────────┘   │
                        │     ┌─────────────────────┐   │
                        │     │  Fresh microVM       │   │
                        │     │  (500mc / 256MB)     │   │
                        │     │  execute code        │   │
                        │     │  ← metrics           │   │
                        │     └─────────────────────┘   │
                        │     ... (4 configs total)      │
                        │                               │
                        │  4. Compute reward             │
                        │  5. Save transition            │
                        └──────────────────────────────┘
                                      │
                              rl_transitions.jsonl
                              rl_code_pairs.jsonl
                                      │
                                      ▼
                            Offline RL Training
                            (PPO / CQL / BCQ etc.)
```

---

## The RL Problem

| RL Component | What it is here |
|---|---|
| **State** | Code features (complexity, line count) + prompt features + recent execution history |
| **Action** | Resource allocation config: CPU millicores + memory MB + timeout |
| **Reward** | +success, +correctness, −resource waste, −latency, −timeout penalty |
| **Next state** | Updated rolling averages of success rate, CPU used, memory used |

The goal: learn a policy that picks the *minimum* resources needed for a task to succeed — don't over-provision (wasteful) or under-provision (causes failures).

---

## Action Space (4 discrete configs)

| Config | CPU | Memory | Timeout | Use case |
|--------|-----|--------|---------|----------|
| 0 | 250 millicores | 128 MB | 5s | Tiny / trivial code |
| 1 | 500 millicores | 256 MB | 10s | Standard code |
| 2 | 1000 millicores | 256 MB | 10s | CPU-heavy code |
| 3 | 1000 millicores | 256 MB | 30s | Long-running code |

---

## File Structure

```
firecracker-rl/
│
├── llm.py                  # MiniMax API calls (via Anthropic SDK compatibility)
├── vm.py                   # Firecracker VM lifecycle: boot, stop, vsock, cgroups
├── actions.py              # RL logic: action configs, reward, state, dataset loading
│
├── collecting_dataset.py   # CLI: collect transitions for full/partial dataset
├── test_one.py             # CLI: test a single random instance end-to-end
├── server.py               # FastAPI REST server wrapping all of the above
│
├── rl_transitions.jsonl    # OUTPUT: (s, a, r, s', done) training data
├── rl_code_pairs.jsonl     # OUTPUT: prompt + generated code + reference code
│
├── datasets/               # MBPP and HumanEval arrow files
├── vmlinux                 # Guest kernel image
├── rootfs.ext4             # Guest root filesystem
├── firecracker             # Firecracker binary
└── .env                    # ANTHROPIC_API_KEY (used as MiniMax key)
```

---

## Data Flow (per instance)

```
1. Load instance
   └── prompt: "Write a function to remove duplicates from a list"
       test_list: ["assert remove_dups([1,1,2]) == [1,2]", ...]

2. Call MiniMax LLM
   └── prompt + "Name the function `remove_dups`. Return only Python code."
       → generated_code: "def remove_dups(lst): ..."

3. Static analysis of generated code
   └── line_count, cyclomatic_complexity, has_recursion, etc.
       → state vector (14 features)

4. For each of 4 resource configs:
   a. Boot a fresh Firecracker microVM with those resource limits
   b. Send generated code via vsock → guest agent executes it
   c. Collect: exit_code, wall_time_ms, cpu_user_ms, mem_peak_kb
   d. Run tests (if available) → tests_passed
   e. Compute reward
   f. Tear down VM completely

5. Save to rl_transitions.jsonl (4 rows) and rl_code_pairs.jsonl (1 row)
```

---

## Output Format

### `rl_transitions.jsonl` — one JSON object per line
```json
{
  "episode_id": "ep_1714000000_11_0",
  "task_id": "11",
  "dataset": "mbpp",
  "state": {
    "prompt_token_count": 18,
    "line_count": 5,
    "cyclomatic_complexity": 2,
    ...
  },
  "action": {
    "cpu_millicores": 250,
    "memory_limit_mb": 128,
    "timeout_ms": 5000
  },
  "execution": {
    "exit_code": 0,
    "wall_time_ms": 43,
    "mem_peak_kb": 8192,
    "tests_passed": true
  },
  "reward": 1.45,
  "reward_components": {
    "r_success": 1.0,
    "r_correctness": 1.0,
    "r_resource": -0.3,
    "r_latency": -0.2,
    "r_timeout": 0.0
  },
  "next_state": { ... }
}
```

### `rl_code_pairs.jsonl` — one JSON object per line
```json
{
  "task_id": "11",
  "dataset": "mbpp",
  "prompt": "Write a function to remove duplicates...",
  "generated_code": "def remove_dups(lst): ...",
  "reference_code": "def remove_dups(lst): return list(set(lst))",
  "tests_passed": true
}
```

---

## Running

### Prerequisites
```bash
# Install dependencies
venv/bin/pip install pyarrow fastapi uvicorn anthropic python-dotenv

# .env must contain:
ANTHROPIC_API_KEY=sk-cp-...   # Your MiniMax API key
```

### Option 1 — Collect dataset via CLI
```bash
# First 100 MBPP instances
venv/bin/python collecting_dataset.py --n 100

# Specific range
venv/bin/python collecting_dataset.py --n 50 --offset 100

# All HumanEval
venv/bin/python collecting_dataset.py --dataset humaneval
```

### Option 2 — Test a single instance
```bash
venv/bin/python test_one.py             # random MBPP
venv/bin/python test_one.py --idx 42    # specific instance
```

### Option 3 — REST server
```bash
venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000

# Run a random instance
curl -X POST http://localhost:8000/run/one

# Run specific instance
curl -X POST http://localhost:8000/run/specific -H "Content-Type: application/json" \
     -d '{"dataset": "mbpp", "idx": 42}'

# Kick off background collection of 100 instances
curl -X POST http://localhost:8000/run/all -H "Content-Type: application/json" \
     -d '{"dataset": "mbpp", "n": 100}'

# Check progress
curl http://localhost:8000/status
```

---

## Why Fresh VM Per Execution?

Each of the 4 resource configs gets a **brand-new Firecracker VM** booted from scratch:

- No filesystem state leaks between executions
- No memory artifacts from previous code runs
- Resource limits are set at VM creation, not patched on a running VM
- Training data accurately reflects what production will look like (where each user request gets its own isolated VM)

This is slower (~4 VM boots per instance) but produces higher-quality training data.

---

## Next Steps (after data collection)

1. Train an offline RL model on `rl_transitions.jsonl` using CQL or IQL
2. Wrap the REST server as a Gymnasium environment (`gym.Env`)
3. Fine-tune online against real VMs with the pre-trained policy as a starting point
