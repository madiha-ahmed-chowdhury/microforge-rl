# VM Spinning — Firecracker MicroVM Lifecycle

This document describes every layer of how Firecracker microVMs are booted, configured, used to execute code, and torn down automatically within the RL training loop. This is the core engineering contribution that makes the system fully automated — no manual VM management, no persistent VMs, no shared state between episodes.

---

## Why Firecracker

Firecracker is a KVM-based Virtual Machine Monitor (VMM) built by AWS for serverless workloads. It boots a minimal Linux kernel with no BIOS, no PCI devices, and no unnecessary drivers.

Key properties that make it suitable here:

| Property | Value |
|---|---|
| Boot time | ~125 ms (kernel + rootfs + Python agent ready) |
| Memory overhead | ~65 MB host RAM per VM (kernel + guest OS baseline) |
| Isolation | Full hardware virtualisation (KVM) — code cannot escape the VM |
| Per-VM config | CPU, memory, kernel args, block device, vsock all set at boot time |
| API | REST over a Unix socket (`api.sock`) — no daemon, no network |
| Footprint | Single statically-linked binary (`./firecracker`) |

Each episode boots a fresh VM, executes code inside it, collects metrics, and destroys the VM. No VM is ever reused across episodes.

---

## Static Assets on Disk

| File | Purpose |
|---|---|
| `vmlinux` | Linux kernel image (x86-64, stripped minimal config) |
| `rootfs.ext4` | Guest root filesystem (Ubuntu 24.04, Python 3, guest agent pre-installed) |
| `firecracker` | Firecracker VMM binary (release v1.15.0-x86_64) |

The rootfs is **shared read-only** in concept but mounted `is_read_only: false` per VM — Firecracker gives each VM its own private copy-on-write overlay so writes in one VM are invisible to others and disappear on teardown.

---

## Per-VM Temp Directory

Every VM gets an isolated temp directory created by `tempfile.mkdtemp(prefix="fc-{vm_id}-")`:

```
/tmp/fc-cfg-42-abc123/
├── vm_config.json     ← JSON config passed to Firecracker at boot
├── vsock.sock         ← Unix socket for vsock proxy (host side)
├── api.sock           ← Firecracker REST API socket
└── firecracker.log    ← Firecracker process log
```

This directory is deleted in full when `stop_vm()` is called, leaving no trace on the host.

---

## VM Configuration at Boot

The config written to `vm_config.json` for each VM:

```json
{
  "boot-source": {
    "kernel_image_path": "./vmlinux",
    "boot_args": "console=ttyS0 reboot=k panic=1 pci=off nomodules ro"
  },
  "drives": [{
    "drive_id":       "rootfs",
    "path_on_host":   "./rootfs.ext4",
    "is_root_device": true,
    "is_read_only":   false
  }],
  "machine-config": {
    "vcpu_count": 1,
    "mem_size_mib": <memory_mb>   ← set per episode by the RL agent
  },
  "vsock": {
    "guest_cid": 3,
    "uds_path": "/tmp/fc-{vm_id}-xxx/vsock.sock"
  }
}
```

**Boot args explained:**
- `console=ttyS0` — serial console (no display needed)
- `reboot=k` — on kernel panic, send SIGKILL instead of rebooting (fast failure)
- `panic=1` — reboot 1 second after panic (safety net)
- `pci=off nomodules` — disable PCI bus and kernel modules (faster boot, smaller attack surface)
- `ro` — mount rootfs read-only at kernel level

`mem_size_mib` is the **only parameter that changes between episodes** at the Firecracker level. CPU throttling is done separately via cgroups after boot (see below).

---

## Boot Sequence Step by Step

```
Host (runner.py)                         Firecracker process           Guest kernel / agent
─────────────────────────────────────────────────────────────────────────────────────────────
boot_vm(vm_id, cpu_mc, memory_mb)
  │
  ├─ start_vm(vm_id, memory_mb)
  │    ├─ mkdtemp()  → /tmp/fc-{vm_id}-xxx/
  │    ├─ write vm_config.json  (mem_size_mib = memory_mb)
  │    ├─ subprocess.Popen(firecracker --api-sock ... --config-file ...)
  │    │                                        ├─ Firecracker starts
  │    │                                        ├─ KVM VM created
  │    │                                        ├─ kernel loaded from vmlinux
  │    │                                        ├─ rootfs.ext4 mounted
  │    │                                        └─ init starts guest agent
  │    └─ _setup_vm_cgroup(vm_id, proc.pid)
  │         ├─ mkdir /sys/fs/cgroup/.../firecracker.slice/fc-{vm_id}/
  │         └─ write proc.pid → cgroup.procs  (VM process enters cgroup)
  │
  ├─ time.sleep(3)   ← allow kernel boot to progress
  │
  ├─ wait_for_agent(vm, timeout=30s)
  │    └─ loop every 1s:
  │         ├─ _vsock_connect(vsock.sock, timeout=2s)
  │         │    ├─ socket.connect(vsock.sock)  [Unix socket → vsock proxy]
  │         │    ├─ send "CONNECT 52\n"         [request port 52 inside guest]
  │         │    └─ wait for "OK ...\n"         [guest agent accepted]
  │         ├─ send {"code": "print('ping')", "timeout": 10}
  │         └─ if response exit_code == 0 → agent is ready, break
  │
  └─ apply_cgroups(vm, cpu_millicores, max(128, memory_mb))
       ├─ cpu.max  = "{quota_us} 100000"   [quota_us = cpu_mc/1000 * 100000]
       └─ memory.max = "{memory_mb * 1024 * 1024}"
```

Total wall time from `boot_vm()` call to ready: **~4–6 seconds** (3s sleep + up to 3s polling).

---

## vsock Communication Protocol

vsock (Virtual Socket) is a host↔guest communication channel built into Firecracker. The host side is exposed as a Unix domain socket (`vsock.sock`); the guest side listens on a port (52).

**Handshake sequence for every message:**

```
Host                                    Guest agent (port 52)
────────────────────────────────────────────────────────────
connect(vsock.sock)                 →
send "CONNECT 52\n"                 →
                                    ← "OK <port>\n"   [vsock proxy response]

send JSON payload + "\n"            →
  {"code": "...",
   "timeout": <seconds>,
   "memory_limit_mb": <mb>,         (optional)
   "cpu_limit_sec": <sec>}          (optional)

                                    ← JSON result + "\n"
  {"exit_code": 0,
   "wall_time_ms": 234,
   "cpu_user_ms": 180,
   "cpu_sys_ms": 12,
   "mem_peak_kb": 18432,
   "stdout": "...",
   "stderr": "",
   "timed_out": false,
   "oom_killed": false}
```

A fresh connection is opened for **every code execution** — no persistent connection state.

---

## Guest Agent

The guest agent is a Python process started automatically by the guest's init system when the VM boots. It listens on vsock port 52 and:

1. Receives a JSON payload over vsock
2. Sets `ulimit -v <memory_limit_mb * 1024>` (virtual memory cap) and `ulimit -t <cpu_limit_sec>` if requested
3. Runs the code in a subprocess with `/usr/bin/time -v` to collect RSS peak
4. Captures `stdout`, `stderr`, `exit_code`, `wall_time_ms`, `cpu_user_ms`, `cpu_sys_ms`, `mem_peak_kb`
5. Detects timeout (`SIGALRM` / process exceeded time) and OOM (`exit_code == -9` when not timed out)
6. Sends back the JSON result and closes the connection

The agent never terminates — it loops waiting for connections until the VM is killed.

---

## Code Execution: stdin Injection

The runner cannot send stdin over vsock as a separate stream. Instead, `run_code_on_vm()` wraps the user code with a stdin injection header:

```python
wrapped = (
    f"import sys, io\n"
    f"sys.stdin = io.TextIOWrapper(io.BytesIO({repr(stdin.encode())}))\n"
    + code
)
```

This replaces `sys.stdin` with an in-memory byte buffer before the user code runs. The user code reads `input()` / `sys.stdin.read()` normally and gets the correct input without any changes.

---

## Resource Enforcement: Two Layers

CPU and memory limits are enforced at **two independent layers**:

### Layer 1 — cgroups v2 (host kernel, applied to Firecracker process)

```
/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/
└── firecracker.slice/
    └── fc-{vm_id}/
        ├── cgroup.procs    ← Firecracker PID lives here
        ├── cpu.max         ← "{quota_us} 100000"
        └── memory.max      ← "{memory_mb * 1024 * 1024}"
```

`cpu.max` uses the Linux CFS bandwidth controller:
```
quota_us = (cpu_millicores / 1000) × 100_000
# e.g. 125mc → quota=12500us per 100000us period → 12.5% of one CPU core
```

`memory.max` caps the total host memory the Firecracker process (and everything inside the VM) can use. When exceeded, the OOM killer fires inside the cgroup.

**Floor:** `host_cgroup_mem = max(128, memory_mb)` — even if the agent selects 64 MB, the cgroup is set to 128 MB minimum because the Firecracker process itself needs ~107 MB of host RAM to run. Below this floor the VM's own process would be OOM-killed before executing any code.

### Layer 2 — ulimit (guest OS, applied inside the VM)

The guest agent sets `ulimit -v` (virtual address space) and `ulimit -t` (CPU time) before running user code. This enforces limits at the guest OS level, independently of the host cgroup.

Both layers working together means:
- The cgroup layer kills the entire VM if it exceeds the host memory cap
- The ulimit layer kills the Python subprocess inside the guest if it exceeds the soft memory cap
- Either path causes `exit_code == -9` without `timed_out` → detected as OOM by the runner

---

## The Two-VM Pattern per Episode

Each training episode uses **two VMs in sequence**:

```
Episode N
│
├── PREP VM  (fixed: 500mc / 512MB / 60s timeout)
│    ├─ boot_vm("prep-N", 500, 512)
│    ├─ run ref_solution or LLM code against generated stdin
│    ├─ capture expected output (oracle)
│    ├─ if LLM code fails → refine once (MAX_REFINEMENT_ATTEMPTS=1)
│    └─ stop_vm(prep_vm)
│
└── CONFIG VM  (dynamic: agent-selected cpu_mc / memory_mb / timeout_ms)
     ├─ boot_vm("cfg-N-0", agent_cpu_mc, agent_memory_mb)
     ├─ run verified code with agent_timeout_ms
     ├─ capture: exit_code, wall_ms, cpu_ms, mem_peak_kb, tests_passed
     ├─ compute res_reward(execution, action)
     └─ stop_vm(config_vm)
```

**Why two VMs:**
- PREP VM always has enough resources to succeed — it establishes ground truth (correct output)
- CONFIG VM tests whether the agent's allocation is sufficient — failure here (OOM, timeout) generates the reward signal that trains the agent
- Separating them prevents PREP failure from contaminating the reward signal

**In `--use-ref` mode** (used for all RL2-only evaluations): the PREP VM still boots and runs the reference solution to capture expected output, then CONFIG VM runs under the agent's allocation. LLM is bypassed entirely.

---

## Execution Metrics Returned

Every `run_code_on_vm()` call returns:

| Field | Source | Description |
|---|---|---|
| `exit_code` | see note below | `0` = success · `1` = runtime error · `-9` = SIGKILL (OOM or ulimit) · `-1` = synthetic sentinel (host code, never guest OS) |
| `timed_out` | guest agent | True if wall time exceeded `timeout_ms` |
| `oom_killed` | guest agent | True if exit_code == -9 and not timed_out |
| `wall_time_ms` | guest agent (`time.time()`) | Total elapsed time including Python startup |
| `cpu_user_ms` | `/usr/bin/time -v` or `resource` | User-mode CPU time consumed |
| `cpu_sys_ms` | `/usr/bin/time -v` or `resource` | Kernel-mode CPU time consumed |
| `mem_peak_kb` | `/proc/self/status` VmPeak or `ru_maxrss` | Peak RSS memory of the Python process |
| `stdout` | subprocess capture | Program output (used for correctness check) |
| `stderr` | subprocess capture | Error output (truncated to 500 chars) |
| `tests_passed` | runner.py | `stdout.strip() == expected.strip()` (None if no expected output) |

---

## Failure Modes and Detection

### exit_code values: what each means and where it is set

| Value | Set by | Meaning |
|---|---|---|
| `0` | Guest OS | Code ran to completion successfully |
| `1` (or any positive) | Guest OS | Python runtime error, syntax error, or `sys.exit(1)` |
| `-9` | Guest OS (SIGKILL) | Kernel killed the process — OOM (`ulimit -v` or cgroup `memory.max`) or CPU time exceeded (`ulimit -t`) |
| `-1` | **Host Python code** (synthetic) | Never produced by Linux. Set in two places only: (1) `vm_runner.py` when `send_code()` raises an exception (vsock drop, socket error); (2) `runner.py` when `boot_vm()` returned `None` and `final_exec` is still `None` after the retry loop |

**Key distinction:** `-9` means the guest kernel killed a running process. `-1` means the host Python code could not reach the guest at all — either the VM never booted, or the vsock connection failed. These are fundamentally different failures that happen at different layers.

### Failure table

| Failure | How it happens | exit_code | Reward |
|---|---|---|---|
| **VM never boots** | Host OOM, Firecracker crash, or `wait_for_agent()` times out after 30 s | `boot_vm()` → `None` → runner sets `exit_code = -1` | −4.0 (Phase 1) / −3.0 if `wall_ms==0` (Phase 2) |
| **64 MB boot failure** | Guest OS baseline ~65 MB: VM starts but agent never responds within 30 s | Same as above — `wait_for_agent()` returns False | Same as above |
| **vsock error mid-run** | Socket drops after VM booted (rare — Firecracker crash during execution) | `send_code()` raises exception → `vm_runner.py` sets `exit_code = -1` | −4.0 / −3.0 |
| **OOM inside guest** | Code exceeds `ulimit -v` or cgroup `memory.max` | Guest OS sends SIGKILL → `exit_code = -9`, `timed_out = False` | `r_oom = −3.0` (plus waste penalties for resources chosen) |
| **CPU timeout** | Code exceeds `ulimit -t` (derived from `timeout_ms`) | Guest OS sends SIGKILL → `exit_code = -9`, `timed_out = True` | `r_timeout = −5.0` |
| **Wall-clock timeout** | Guest agent's timer fires before code finishes | `timed_out = True`, `exit_code` may be non-zero | `r_timeout = −5.0` |
| **Wrong answer** | Code runs but `stdout ≠ expected` | `exit_code = 0`, `tests_passed = False` | Included in `llm_reward` (online RL) |
| **Runtime error** | Syntax error, uncaught exception | `exit_code = 1`, stdout empty | Treated as failed execution |

The 64 MB boot failure threshold exists because the guest OS + Python runtime baseline occupies ~65 MB, leaving no room for the guest agent to start within the 30-second polling window.

---

## Why exit_code == -9 (True OOM) Is Rare in Practice

Across all training runs (6,421 total transitions: SAC-Shared 1,454 + SAC-Factored 1,300 + DQN 1,200 + PPO 2,467), **exit_code == -9 never appears once**. Every infrastructure failure shows up as exit_code == -1 instead. This is counterintuitive — you would expect some OOM kills if the agent occasionally allocates too little memory. The reason is a hidden third cause of exit_code == -1 that is easy to miss.

### The stdin embedding problem

The runner does not stream stdin to the guest over a file descriptor. Instead, it bakes the entire stdin into the Python source as a bytes literal:

```python
# vm_runner.py line 28
wrapped = f"import sys, io\nsys.stdin = io.TextIOWrapper(io.BytesIO({repr(stdin.encode())}))\n{code}"
```

This `wrapped` string — containing the full stdin as `b'...'` — is JSON-serialized and sent as a single vsock payload. The guest Python agent then has to **compile** that literal before execution begins. For small inputs this is fine. But stress-test generators produce large inputs:

| Problem | Generator stdin size |
|---|---|
| cc_316_B2. EKG | 390 KB |
| cc_1407_E. Egor in the Republic of Dagestan | 7.5 MB |

A 7.5 MB stdin becomes a 7.5 MB bytes literal in the source. Python must hold both the source text and the compiled bytes object in memory simultaneously during compilation — roughly doubling the footprint. If the VM's cgroup memory limit is tight, the guest Python process is OOM-killed **during compilation**, before the user code runs even one line. This drops the vsock connection. `send_code()` raises a `ConnectionResetError`, caught by `vm_runner.py`, and returns `exit_code = -1`.

This is indistinguishable from a boot failure in the logged data because neither `stderr` nor the failure path is stored in `transitions.jsonl` — only the six scalar fields (`exit_code`, `timed_out`, `wall_time_ms`, `cpu_user_ms`, `mem_peak_kb`, `tests_passed`) are saved per transition.

### What the data shows

Across `transitions.jsonl` (SAC-Shared, 1,454 episodes), exit_code == -1 cases broken down by memory allocation:

| Memory allocated | Count | Likely cause |
|---|---|---|
| 32–64 MB | 75 | VM boot failure (below guest OS baseline) |
| 80–128 MB | 108 | Boot failure (marginal) or compilation OOM on medium inputs |
| 160–256 MB | 20 | Compilation OOM on large inputs — VM booted fine |
| 320–512 MB | 27 | Compilation OOM on very large inputs — definitively not a boot failure |

The 320–512 MB cases prove the point: those VMs have more than enough memory to boot Firecracker. They fail because a large stdin payload overwhelmed the guest during the compilation phase, not during code execution. This is why true OOM (exit_code == -9, which requires the code to actually run and exceed its memory limit) is zero — the failures happen earlier, during the compilation of the embedded input.

**Summary:** The agent is not "learning to avoid OOM." The OOM signal never reaches the guest execution phase for large-input problems because the vsock payload itself triggers a pre-execution crash that masquerades as a connection failure (exit_code == -1).

---

## Python Startup Overhead

A key calibration measurement: **how long does Python take to start inside a Firecracker VM?**

Timing measurements across 5 programs × 5 runs at 128 MB / 100 mc:

| Program | Min (ms) | Mean (ms) | Max (ms) |
|---|---|---|---|
| Trivial (`a = 1+1; print(a)`) | ~50 | ~80 | ~177 |
| Imports only (sys, math, collections) | ~80 | ~120 | ~200 |
| Read + sum (stdin I/O) | ~90 | ~130 | ~210 |
| Typical easy CP (Counter, defaultdict) | ~100 | ~150 | ~230 |
| Sort + binary search (bisect) | ~110 | ~160 | ~250 |

**Conclusion:** Python interpreter startup inside the guest adds 50–250 ms to wall time regardless of what the code does. This is why:
- `TIMEOUT_STARTUP_OVERHEAD_MS = 200` is set as a floor in reward computation
- `min_viable_tms = max(200, wall_ms * 2.0)` prevents rewarding sub-200ms timeout as "tight"
- Timeout bins below 200 ms are not viable even for trivial code

---

## Automation: Full Episode Lifecycle

The entire lifecycle — from problem selection to reward computation — runs without any human intervention:

```
runner.py (main loop)
│
│  ┌─────────────────────────────────────────────────────────────────┐
│  │  Episode N                                                       │
│  │                                                                  │
│  │  1. problem = random.choice(pool)  ← from cc_pool_cache.json    │
│  │                                                                  │
│  │  2. [--use-ref skips this] LLM bandit selects model             │
│  │     → generate_code(description, stdin, expected, tier)          │
│  │                                                                  │
│  │  3. PREP VM boots at 500mc/512MB                                 │
│  │     → run_code_on_vm(prep_vm, code, stdin, 60000ms)              │
│  │     → capture expected output, test correctness                  │
│  │     → stop_vm(prep_vm)                                           │
│  │                                                                  │
│  │  4. static_analyse(code)  → 9 or 18 code features               │
│  │     build_state_vec(features, rolling, scaler)                   │
│  │                                                                  │
│  │  5. res_agent.select_action(state)                               │
│  │     → (cpu_idx, mem_idx, tms_idx) → (cpu_mc, mem_mb, tms_ms)    │
│  │                                                                  │
│  │  6. CONFIG VM boots at agent_cpu_mc / agent_mem_mb              │
│  │     → run_code_on_vm(cfg_vm, code, stdin, agent_tms_ms)          │
│  │     → collect exit_code, wall_ms, mem_peak_kb, tests_passed      │
│  │     → stop_vm(cfg_vm)                                            │
│  │                                                                  │
│  │  7. res_reward = compute_res_reward(execution, action)           │
│  │     update_rolling(rolling, execution)                           │
│  │                                                                  │
│  │  8. [training mode only]                                         │
│  │     replay_buffer.push(state, action, reward, next_state)        │
│  │     res_agent.update(buffer.sample(256))                         │
│  │                                                                  │
│  │  9. log transition to transitions.jsonl                          │
│  └─────────────────────────────────────────────────────────────────┘
│
└── repeat for args.episodes
```

**What is automated:**
- VM provisioning (boot, wait, apply limits, execute, tear down)
- Resource selection (RL agent picks CPU/memory/timeout per problem)
- Input generation (test_case_generator runs, feeds ref_solution for expected output)
- Correctness checking (stdout comparison, no human labels)
- Reward computation (formulaic from execution metrics)
- Agent updates (replay buffer sampling, gradient steps)
- Checkpointing (every 10 episodes)
- Problem pool management (random sampling, no curriculum needed)

**What requires human intervention:**
- Initial rootfs build (one-time: build Ubuntu image with Python + guest agent)
- Pool cache construction (`cc_pool_cache.json` fetched from HuggingFace once)
- Hyperparameter selection (lr, γ, τ, entropy target — set once per agent)

---

## cgroups Hierarchy

```
/sys/fs/cgroup/
└── user.slice/
    └── user-{uid}.slice/
        └── user@{uid}.service/          ← user delegated cgroup root
            └── firecracker.slice/       ← created once per host session
                ├── fc-prep-0/           ← PREP VM for episode 0
                ├── fc-cfg-0-0/          ← CONFIG VM for episode 0
                ├── fc-prep-1/           ← PREP VM for episode 1
                └── fc-cfg-1-0/          ← CONFIG VM for episode 1
```

Each VM cgroup is created at boot and deleted at `stop_vm()`. Parent dirs (`firecracker.slice`) persist across episodes and are cleaned up only when empty.

---

## FastAPI REST Layer (data collection phase)

During the offline data collection phase, a FastAPI server (`server.py`) wrapped the VM lifecycle:

| Endpoint | What it does |
|---|---|
| `POST /vm/create` | Boot a VM, return `vm_id` |
| `DELETE /vm/{vm_id}` | `stop_vm()` and remove from registry |
| `POST /vm/{vm_id}/execute` | `send_code()` — run code, return metrics |
| `PATCH /vm/{vm_id}/resources` | `apply_cgroups()` — change CPU/memory limits live |
| `GET /vm/{vm_id}/metrics` | Last execution result |
| `GET /vm/{vm_id}/status` | PID, alive, uptime |
| `POST /run/one` | Full pipeline: boot VM → LLM → execute → log |
| `POST /run/all` | Background loop over full dataset |
| `GET /status` | Collection progress |

In the online RL phase, this REST layer is bypassed — `boot_vm()`, `run_code_on_vm()`, and `stop_vm()` are called directly from `runner.py`.
