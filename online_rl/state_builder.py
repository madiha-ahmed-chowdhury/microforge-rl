import ast as _ast

from online_rl.config import (
    FEATURE_COLS,
    CPU_MIN, CPU_MAX,
    MEM_MIN, MEM_MAX,
    TMS_MIN, TMS_MAX,
    LLM_TIERS, LLM_THRESHOLDS,
    OPUS_WARMUP_BLOCK, OPUS_MAX_PCT,
)


def static_analyse(code: str) -> dict:
    features = {
        "cyclomatic_complexity": 1,
        "max_loop_depth":        0,
        "ast_node_count":        0,
        "has_recursion":         0,
        "has_external_calls":    0,
    }
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return features

    features["ast_node_count"] = sum(1 for _ in _ast.walk(tree))

    branch_types = (
        _ast.If, _ast.For, _ast.While, _ast.ExceptHandler, _ast.With, _ast.Assert,
        _ast.ListComp, _ast.DictComp, _ast.SetComp, _ast.GeneratorExp,
    )
    features["cyclomatic_complexity"] = 1 + sum(
        1 for n in _ast.walk(tree) if isinstance(n, branch_types)
    )

    def _max_depth(node, d=0):
        if isinstance(node, (_ast.For, _ast.While)):
            d += 1
        return max([d] + [_max_depth(c, d) for c in _ast.iter_child_nodes(node)])

    features["max_loop_depth"] = _max_depth(tree)

    defined = {n.name for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef)}
    called  = {
        n.func.id for n in _ast.walk(tree)
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)
    }
    features["has_recursion"]      = int(bool(defined & called))
    features["has_external_calls"] = int(
        any(k in code.lower() for k in ("subprocess", "requests", "urllib", "open(", "socket", "http"))
    )
    return features


def build_state_vec(problem_state: dict, code_features: dict, scaler) -> list:
    merged = dict(problem_state)
    for k, v in code_features.items():
        if k in FEATURE_COLS:
            merged[k] = v
    raw = [float(merged.get(k, 0)) for k in FEATURE_COLS]
    if scaler is not None:
        import numpy as np
        raw = scaler.transform([raw])[0].tolist()
    return raw


def update_rolling(state: list, execution: dict) -> list:
    next_s  = list(state)
    idx_map = {k: i for i, k in enumerate(FEATURE_COLS)}
    success = 1.0 if execution.get("exit_code") == 0 else 0.0

    if "recent_success_rate" in idx_map:
        i = idx_map["recent_success_rate"]
        next_s[i] = round(success * 0.1 + state[i] * 0.9, 4)
    if "recent_mean_cpu_used" in idx_map:
        i = idx_map["recent_mean_cpu_used"]
        next_s[i] = round(execution.get("cpu_user_ms", 0) * 0.1 + state[i] * 0.9, 2)
    if "recent_mean_mem_used" in idx_map:
        i = idx_map["recent_mean_mem_used"]
        next_s[i] = round(execution.get("mem_peak_kb", 0) * 0.1 + state[i] * 0.9, 2)
    return next_s


def scale_action(a) -> dict:
    cpu = int((a[0] + 1) / 2 * (CPU_MAX - CPU_MIN) + CPU_MIN)
    mem = int((a[1] + 1) / 2 * (MEM_MAX - MEM_MIN) + MEM_MIN)
    tms = int((a[2] + 1) / 2 * (TMS_MAX - TMS_MIN) + TMS_MIN)
    return {
        "cpu_millicores": max(CPU_MIN, min(CPU_MAX, cpu)),
        "memory_mb":      max(MEM_MIN, min(MEM_MAX, mem)),
        "timeout_ms":     max(TMS_MIN, min(TMS_MAX, tms)),
    }


def select_llm_tier(a4: float, episode: int, opus_used: int, total_eps: int) -> str:
    if a4 < LLM_THRESHOLDS[0]:
        tier = "laguna"
    elif a4 < LLM_THRESHOLDS[1]:
        tier = "sonnet"
    else:
        tier = "opus"

    if tier == "opus" and episode < OPUS_WARMUP_BLOCK:
        tier = "sonnet"
    if tier == "opus" and total_eps > 0 and opus_used / total_eps >= OPUS_MAX_PCT:
        tier = "sonnet"

    return tier
