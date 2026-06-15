#!/usr/bin/env python3
"""
eval_results.py

Reads online_rl/results/eval_transitions.jsonl and prints RL1 + RL2 result tables
per eval_category (high_memory, easy, medium, hard).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

TRANSITIONS = "online_rl/results/eval_transitions.jsonl"

def load(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(v, n=3) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{n}f}"


def print_table(headers: list[str], rows: list[list], col_width: int = 14) -> None:
    line = "  ".join(h.ljust(col_width) for h in headers)
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(c).ljust(col_width) for c in row))


CATS = ["high_memory", "easy", "medium", "hard", "OVERALL"]


def main() -> None:
    path = Path(TRANSITIONS)
    if not path.exists():
        print(f"[eval_results] {TRANSITIONS} not found — run the eval first.")
        sys.exit(1)

    rows = load(str(path))
    if not rows:
        print("[eval_results] file is empty.")
        sys.exit(1)

    print(f"[eval_results] loaded {len(rows)} transitions\n")

    # Group by category
    by_cat: dict[str, list] = defaultdict(list)
    for r in rows:
        cat = r.get("eval_category") or r.get("cf_tags_cat") or "unknown"
        by_cat[cat].append(r)
    by_cat["OVERALL"] = rows

    # ── RL1: Bandit / LLM ─────────────────────────────────────────────────────
    print("=" * 70)
    print("RL1  —  Bandit / LLM Agent")
    print("=" * 70)

    rl1_rows = []
    for cat in CATS:
        grp = by_cat.get(cat, [])
        if not grp:
            continue
        n = len(grp)
        passed  = sum(1 for r in grp if r.get("execution", {}).get("tests_passed"))
        rewards = [r["llm_reward"] for r in grp if "llm_reward" in r]
        avg_rew = sum(rewards) / len(rewards) if rewards else None
        pass_rt = passed / n * 100

        model_counts: dict[str, int] = defaultdict(int)
        for r in grp:
            model_counts[r.get("llm_model", "?")[:20]] += 1

        top_model = max(model_counts, key=model_counts.__getitem__) if model_counts else "?"
        rl1_rows.append([
            cat[:14],
            n,
            f"{pass_rt:.1f}%",
            fmt(avg_rew),
            top_model[:18],
        ])

    print_table(
        ["category", "n", "pass%", "avg_llm_rew", "top_model"],
        rl1_rows,
        col_width=16,
    )

    # Model arm distribution
    print("\nModel arm distribution (all episodes):")
    model_all: dict[str, int] = defaultdict(int)
    for r in rows:
        model_all[r.get("llm_model", "?")[:30]] += 1
    for m, cnt in sorted(model_all.items(), key=lambda x: -x[1]):
        print(f"  {m:<35} {cnt:>4}  ({cnt/len(rows)*100:.1f}%)")

    # ── RL2: SAC / Resource ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RL2  —  SAC / Resource Agent")
    print("=" * 70)

    rl2_rows = []
    for cat in CATS:
        grp = by_cat.get(cat, [])
        if not grp:
            continue
        n = len(grp)
        rewards = [r["res_reward"] for r in grp if "res_reward" in r]
        avg_rew = sum(rewards) / len(rewards) if rewards else None

        # OOM / timeout rates
        ooms     = sum(1 for r in grp if r.get("execution", {}).get("mem_peak_kb", 0) == 0
                       and not r.get("execution", {}).get("tests_passed"))
        timeouts = sum(1 for r in grp if r.get("execution", {}).get("timed_out"))
        rl2_rows.append([
            cat[:14],
            n,
            fmt(avg_rew),
            f"{timeouts}/{n}",
        ])

    print_table(
        ["category", "n", "avg_res_rew", "timeouts"],
        rl2_rows,
        col_width=16,
    )

    # Resource bin distribution
    print("\nResource bin distribution (all episodes):")
    cpu_counts: dict = defaultdict(int)
    mem_counts: dict = defaultdict(int)
    tmo_counts: dict = defaultdict(int)
    for r in rows:
        act = r.get("action", {})
        cpu_counts[act.get("cpu_idx", "?")] += 1
        mem_counts[act.get("mem_idx", "?")] += 1
        tmo_counts[act.get("timeout_idx", "?")] += 1

    print("  CPU idx:", dict(sorted(cpu_counts.items())))
    print("  Mem idx:", dict(sorted(mem_counts.items())))
    print("  TmO idx:", dict(sorted(tmo_counts.items())))

    # Wall time stats
    times = [r.get("execution", {}).get("wall_time_ms", 0) for r in rows]
    if times:
        print(f"\nWall time (ms): min={min(times):.0f}  median={sorted(times)[len(times)//2]:.0f}  max={max(times):.0f}")

    # avg10 trend
    avg10s = [r.get("avg10") for r in rows if r.get("avg10") is not None]
    if avg10s:
        print(f"avg10 res_reward: start={avg10s[0]:.3f}  end={avg10s[-1]:.3f}")


if __name__ == "__main__":
    main()
