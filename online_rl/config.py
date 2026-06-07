FEATURE_COLS = [
    "prompt_token_count",
    "prompt_complexity_score",
    "task_type",
    "has_loops_hint",
    "has_io_hint",
    "example_count",
    "line_count",
    "cyclomatic_complexity",
    "ast_node_count",
    "has_recursion",
    "has_external_calls",
    "max_loop_depth",
    "estimated_complexity",
    "host_cpu_load_1m",
    "host_mem_available_mb",
    "queue_depth",
    "recent_success_rate",
    "recent_mean_cpu_used",
    "recent_mean_mem_used",
]

# Action ranges — fully continuous, no rounding
CPU_MIN, CPU_MAX = 50, 500      # millicores
MEM_MIN, MEM_MAX = 48, 256      # MB — 48MB minimum: Python needs ~40MB virtual memory overhead
TMS_MIN, TMS_MAX = 1000, 10000  # ms

PREP_CONFIG = {
    "cpu_millicores": 500,
    "memory_mb":      256,
    "timeout_ms":     30000,
}

# Agent 1 — LLM Selector: sees problem features, picks which LLM to call
LLM_SAC_CONFIG = {
    "lr":                   3e-4,
    "gamma":                0.99,
    "tau":                  0.005,
    "batch_size":           128,
    "buffer_size":          5000,
    "warmup":               100,
    "target_entropy_ratio": 0.98,
    "state_dim":            19,
    "action_dim":           1,
}

# Agent 2 — Resource Allocator: sees problem + generated code features, picks cpu/mem/timeout
RES_SAC_CONFIG = {
    "lr":                   3e-4,
    "gamma":                0.99,
    "tau":                  0.005,
    "batch_size":           256,
    "buffer_size":          5000,
    "warmup":               200,
    "target_entropy_ratio": 0.98,
    "state_dim":            19,
    "action_dim":           3,
}

SAC_CONFIG = RES_SAC_CONFIG  # keep backward compat

REWARD_CONFIG = {
    "r_timeout":    -5.0,
    "r_oom":        -3.0,
    "cpu_waste_w":  -0.5,
    "mem_waste_w":  -2.0,
    "latency_w":    -0.2,
    "latency_base": 500,
    "llm_costs":    {"laguna": 0.0, "haiku": -0.05, "sonnet": -0.3, "opus": -1.0},
}

# LLM tier selection — 4th action dimension thresholded
# < 0.5  → laguna  (free, fast)
# 0.5–0.9 → sonnet  (paid, strong)
# > 0.9  → opus    (expensive, rare)
LLM_TIERS      = ["laguna", "sonnet", "opus"]
LLM_THRESHOLDS = [0.5, 0.9]

# Hard caps: no Opus for first N episodes, and never more than X% of total
OPUS_WARMUP_BLOCK  = 200    # no Opus until episode 200
OPUS_MAX_PCT       = 0.02   # Opus capped at 2% of all episodes after warmup

PATHS = {
    "inputs":          "online_rl/inputs.json",
    "scaler":          "collected/scaler.pkl",
    "checkpoints":     "online_rl/checkpoints/",
    "results":         "online_rl/results/",
    "curve":           "online_rl/results/curve.jsonl",
    "transitions":     "online_rl/results/transitions.jsonl",
    "code_pairs":      "collected/effibench_code_pairs_dedup.jsonl",
    "raw_transitions": "collected/effibench_transitions.jsonl",
}

VSOCK = {
    "AF_VSOCK": 40,
    "port":      52,
    "guest_cid": 3,
}

MAX_RESOURCE_RETRIES     = 3
MAX_REFINEMENT_ATTEMPTS  = 2
