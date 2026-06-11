# Firecracker microVM Memory Findings

## Test Program

```python
a = 1 + 1
print(a)
```

## Results by Allocation

| Allocation | Guest sees | Used | Free | agent.py RSS | Wall time | Status |
|---|---|---|---|---|---|---|
| 32MB | — | — | — | — | — | VM boot failed |
| 64MB | 41.8MB | 38.9MB | 2.9MB | 7MB | 2344ms | Works but very slow |
| 128MB | 107MB | 45MB | 62MB | 11.4MB | 93ms | Comfortable |

## Memory Breakdown per Layer (128MB allocation)

```
128MB allocated by host
│
├── Firecracker VMM overhead (host-side, never reaches guest)
│     ~21MB  (128MB - 107MB guest MemTotal)
│
└── Guest RAM: 107MB
      ├── Guest Linux kernel (code, slab, page tables, buffers)
      │     ~33MB
      ├── agent.py process (persistent vsock server)
      │     ~11MB RSS
      ├── User code subprocess (bash → python3 → script)
      │     ~13MB  ← this is what mem_peak_kb reports
      └── Free
            ~62MB
```

## What mem_peak_kb Actually Measures

`mem_peak_kb` in execution results comes from `resource.getrusage(RUSAGE_CHILDREN)` inside agent.py.
It measures **only the user code subprocess** (bash + python3). It does NOT include:
- Firecracker VMM overhead
- Guest Linux kernel memory
- agent.py's own memory

So `mem_peak_kb ≈ 13MB` for a trivial Python script is the Python interpreter baseline,
not total VM memory usage.

## Practical Minimums

- **32MB** — unusable, VM cannot boot (kernel + agent alone need ~44MB)
- **64MB** — technically works but only 2.9MB free, execution is 25× slower (2344ms vs 93ms)
- **128MB** — comfortable minimum, 62MB free for user code

## Implication for Reward Function

The memory waste penalty in the current reward function penalises allocations above `mem_peak_kb`.
But `mem_peak_kb` ≈ 13MB for almost all Python programs regardless of their logic,
because the Python interpreter itself costs ~13MB.

The true overhead the SAC agent cannot reduce:
- Firecracker VMM: ~21MB (not in guest)
- Guest kernel: ~33MB
- agent.py: ~11MB
- Python interpreter baseline: ~13MB
- **Total fixed overhead: ~78MB**

Even with the tightest allocation (128MB), only ~50MB is actually available for user logic.

### Proposed Fix: Baseline Subtraction in Reward

Instead of penalising total `mem_peak_kb`, subtract the Python baseline (~13MB)
before computing waste — this way SAC is only penalised for memory *above* what Python itself needs:

```python
PYTHON_BASELINE_KB = 13 * 1024

mem_used_kb  = max(0, execution.get("mem_peak_kb", 0) - PYTHON_BASELINE_KB)
mem_alloc_kb = max(0, action["memory_mb"] * 1024 - PYTHON_BASELINE_KB)
mem_waste    = max(0, (mem_alloc_kb - mem_used_kb) / max(mem_alloc_kb, 1))
```

## Files

- `program.py` — test program (`a = 1 + 1; print(a)`)
- `run_test.py` — boots VMs at 32/64/128MB and measures execution
- `measure_overhead.py` — reads `/proc/meminfo` and `/proc/self/status` from inside guest
- `result.json` — raw results from run_test.py
