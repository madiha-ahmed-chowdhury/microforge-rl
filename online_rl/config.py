import math as _math

# ── Discrete resource bins ─────────────────────────────────────────────────
CPU_BINS     = [50, 75, 100, 125, 150, 175, 200, 300, 500]
MEMORY_BINS  = [64, 80, 96, 112, 128, 160, 192, 256, 320, 512]
TIMEOUT_BINS = [200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 5000, 8000, 10000, 15000, 20000, 30000]

N_CPU     = 9   # len(CPU_BINS)
N_MEMORY  = 10  # len(MEMORY_BINS)
N_TIMEOUT = 15 # len(TIMEOUT_BINS)

FIXED_OVERHEAD_MB = 65  # guest OS + Python runtime baseline

SAC_FEATURE_COLS = [
    "cyclomatic_complexity",
    "max_loop_depth",
    "estimated_complexity",
    "has_recursion",
    "ast_node_count",
    "line_count",
    "recent_success_rate",
    "recent_mean_cpu_used",
    "recent_mean_mem_used",
]

# Action ranges — fully continuous, no rounding
CPU_MIN, CPU_MAX = 50, 500      # millicores
MEM_MIN, MEM_MAX = 64, 512      # MB — 64MB minimum (matches lowest MEMORY_BIN)
TMS_MIN, TMS_MAX = 200, 10000  # ms

PREP_CONFIG = {
    "cpu_millicores": 500,
    "memory_mb":      512,
    "timeout_ms":     60000,
}

# Agent 2 — Resource Allocator: discrete SAC over CPU/memory/timeout bins
RES_SAC_CONFIG = {
    "lr_actor":             3e-4,
    "lr_critic":            5e-4,
    "lr_alpha":             3e-4,
    "gamma":                0.99,
    "tau":                  0.005,
    "batch_size":           256,
    "buffer_size":          20000,
    "warmup":               200,
    "target_entropy":       -(_math.log(N_CPU) + _math.log(N_MEMORY) + _math.log(N_TIMEOUT)) * 0.98,
    "state_dim":            9,
}

SAC_CONFIG = RES_SAC_CONFIG  # keep backward compat

REWARD_CONFIG = {
    "r_timeout":    -5.0,
    "r_oom":        -3.0,
    "cpu_waste_w":  -0.5,
    "mem_waste_w":  -2.0,
    "latency_w":    -0.2,
    "latency_base": 500,
    "llm_costs":    {"free": 0.0, "haiku": -0.05, "sonnet": -0.3, "opus": -1.0},
}

# LLM tier selection was previously a 4th action dimension; now handled by LLMBandit.
# LLM_TIERS      = ["free", "sonnet", "opus"]
# LLM_THRESHOLDS = [0.5, 0.9]
# OPUS_WARMUP_BLOCK  = 200
# OPUS_MAX_PCT       = 0.02

PATHS = {
    "inputs":          "online_rl/inputs.json",
    "scaler":          "collected/scaler.pkl",
    "checkpoints":     "online_rl/checkpoints/",
    "results":         "online_rl/results/",
    "curve":           "online_rl/results/curve.jsonl",
    "transitions":     "online_rl/results/transitions.jsonl",
    "code_pairs":      "collected/effibench_code_pairs_dedup.jsonl",
    "raw_transitions": "collected/effibench_transitions.jsonl",
    "code_cache":      "online_rl/results/code_cache.json",
    "online_scaler":   "online_rl/results/online_scaler.pkl",
}

VSOCK = {
    "AF_VSOCK": 40,
    "port":      52,
    "guest_cid": 3,
}

VM_BOOT_TIMEOUT             = 30   # seconds for wait_for_agent polling
TIMEOUT_STARTUP_OVERHEAD_MS = 200  # Python interpreter startup floor for reward
MAX_RESOURCE_RETRIES        = 3
MAX_REFINEMENT_ATTEMPTS  = 1
MAX_CACHE_RETRY_ATTEMPTS = 3  # LLM re-calls before giving up on a failing cached problem
