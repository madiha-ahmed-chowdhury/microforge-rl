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
        # memory-signal features
        "uses_defaultdict":    0,
        "uses_deque":          0,
        "uses_heapq":          0,
        "has_array_mult":      0,  # [x]*N — segment trees, BIT, large arrays
        "has_collection_list": 0,  # [[] for _ in range(n)] — adjacency/set graphs
        # timeout-signal features
        "uses_itertools":      0,  # permutations/combinations → exponential time
        "has_lru_cache":       0,  # memoization present → recursive but cached
        "sort_call_count":     0,  # number of sort/sorted calls → O(n log n) passes
        "has_while_true":      0,  # while True → unbounded loop risk
    }
    features["line_count"] = len([l for l in code.splitlines() if l.strip()])

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return features

    features["ast_node_count"] = sum(1 for _ in _ast.walk(tree))

    # memory-signal detection via imports
    imported_names = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                imported_names.add(alias.name.split('.')[0])
        elif isinstance(node, _ast.ImportFrom):
            if node.module:
                imported_names.add(node.module.split('.')[0])
            for alias in node.names:
                imported_names.add(alias.name)

    features["uses_heapq"]       = int("heapq" in imported_names)
    features["uses_defaultdict"]  = int("defaultdict" in imported_names or "defaultdict" in code)
    features["uses_deque"]        = int("deque" in imported_names or "deque" in code)

    # [x]*N — covers [0]*n, [0]*(4*n+10) (segment tree), [None]*(n+1) (BIT)
    features["has_array_mult"] = int(any(
        isinstance(node, _ast.BinOp) and
        isinstance(node.op, _ast.Mult) and
        (isinstance(node.left, _ast.List) or isinstance(node.right, _ast.List))
        for node in _ast.walk(tree)
    ))

    # [[] for _ in range(n)], [{x} for _ in range(n)] — adjacency/set graphs
    # also catches sys.setrecursionlimit → deep tree recursion
    features["has_collection_list"] = int(
        any(
            isinstance(node, _ast.ListComp) and
            isinstance(node.elt, (_ast.List, _ast.Set, _ast.Dict))
            for node in _ast.walk(tree)
        ) or any(
            isinstance(node, _ast.Call) and
            isinstance(node.func, _ast.Attribute) and
            node.func.attr == "setrecursionlimit"
            for node in _ast.walk(tree)
        )
    )

    # timeout-signal detection
    features["uses_itertools"] = int("itertools" in imported_names)

    features["has_lru_cache"] = int(
        "lru_cache" in imported_names or
        "cache" in imported_names or
        any(
            isinstance(node, _ast.Call) and
            isinstance(node.func, _ast.Attribute) and
            node.func.attr in ("lru_cache", "cache")
            for node in _ast.walk(tree)
        )
    )

    features["sort_call_count"] = sum(
        1 for node in _ast.walk(tree)
        if isinstance(node, _ast.Call) and (
            (isinstance(node.func, _ast.Name) and node.func.id == "sorted") or
            (isinstance(node.func, _ast.Attribute) and node.func.attr == "sort")
        )
    )

    features["has_while_true"] = int(any(
        isinstance(node, _ast.While) and (
            (isinstance(node.test, _ast.Constant) and node.test.value is True) or
            (isinstance(node.test, _ast.NameConstant) and node.test.value is True)
        )
        for node in _ast.walk(tree)
    ))

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


_RAW_SCALES = [50.0, 10.0, 5.0, 1.0, 1000.0, 200.0, 1.0, 200.0, 50000.0,
               1.0, 1.0, 1.0, 1.0, 1.0,   # memory-signal features (binary)
               1.0, 1.0, 10.0, 1.0]        # timeout-signal features

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
