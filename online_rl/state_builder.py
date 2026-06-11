import ast as _ast

from online_rl.config import (
    SAC_FEATURE_COLS,
    CPU_MIN, CPU_MAX,
    MEM_MIN, MEM_MAX,
    TMS_MIN, TMS_MAX,
)


def static_analyse(code: str) -> dict:
    features = {
        "cyclomatic_complexity": 1,
        "max_loop_depth":        0,
        "ast_node_count":        0,
        "has_recursion":         0,
        "has_external_calls":    0,
        "line_count":            0,
        "estimated_complexity":  0,
    }
    features["line_count"] = len([l for l in code.splitlines() if l.strip()])

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
    # has_external_calls not in SAC_FEATURE_COLS — computed but unused
    # features["has_external_calls"] = int(
    #     any(k in code.lower() for k in ("subprocess", "requests", "urllib", "open(", "socket", "http"))
    # )

    cc = features["cyclomatic_complexity"]
    ld = features["max_loop_depth"]
    if cc > 15 or ld >= 4:
        features["estimated_complexity"] = 3
    elif cc > 8 or ld >= 3:
        features["estimated_complexity"] = 2
    elif cc > 3 or ld >= 2:
        features["estimated_complexity"] = 1
    else:
        features["estimated_complexity"] = 0

    return features


_RAW_SCALES = [50.0, 10.0, 5.0, 1.0, 1000.0, 200.0, 1.0, 200.0, 50000.0]

def build_state_vec(code_features: dict, rolling: dict, scaler) -> list:
    merged = {}
    merged.update(code_features)
    merged.update(rolling)
    raw = [float(merged.get(k, 0.0)) for k in SAC_FEATURE_COLS]
    if scaler is not None:
        import numpy as np
        raw = scaler.transform([raw])[0].tolist()
    else:
        raw = [v / s for v, s in zip(raw, _RAW_SCALES)]
    return raw


def update_rolling(rolling: dict, execution: dict) -> dict:
    success = 1.0 if (
        not execution.get("timed_out", False) and
        not execution.get("oom_killed", False) and
        execution.get("exit_code") != -9
    ) else 0.0

    return {
        "recent_success_rate":  round(
            success * 0.1 + rolling["recent_success_rate"] * 0.9, 4),
        "recent_mean_cpu_used": round(
            execution.get("cpu_user_ms", 0) * 0.1 + rolling["recent_mean_cpu_used"] * 0.9, 2),
        "recent_mean_mem_used": round(
            execution.get("mem_peak_kb", 0) * 0.1 + rolling["recent_mean_mem_used"] * 0.9, 2),
    }


# scale_action: continuous tanh → resource values, used by old SACAgent (continuous).
# DiscreteSACAgent returns bin indices directly — this is unused.
# def scale_action(a) -> dict:
#     cpu = int((a[0] + 1) / 2 * (CPU_MAX - CPU_MIN) + CPU_MIN)
#     mem = int((a[1] + 1) / 2 * (MEM_MAX - MEM_MIN) + MEM_MIN)
#     tms = int((a[2] + 1) / 2 * (TMS_MAX - TMS_MIN) + TMS_MIN)
#     return {
#         "cpu_millicores": max(CPU_MIN, min(CPU_MAX, cpu)),
#         "memory_mb":      max(MEM_MIN, min(MEM_MAX, mem)),
#         "timeout_ms":     max(TMS_MIN, min(TMS_MAX, tms)),
#     }
