import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from online_rl.config import PATHS, FEATURE_COLS


def _load_state_map() -> dict:
    state_map = {}
    try:
        with open(PATHS["raw_transitions"]) as f:
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line)
                tid = t.get("task_id")
                if tid and tid not in state_map:
                    state_map[tid] = t.get("state", {})
    except FileNotFoundError:
        pass
    return state_map


def _build_default_state(row: dict) -> dict:
    from actions import static_analyse, build_state
    code = row.get("code", "")
    cf   = static_analyse(code)
    return build_state(row.get("prompt", ""), cf, row.get("task_type", 3))


def build_from_hf(heavy_only: bool = False) -> list:
    from actions import _load_effibench_hf
    state_map = _load_state_map()

    dataset = "effibench_large" if heavy_only else "effibench"
    rows    = _load_effibench_hf(large=heavy_only)
    print(f"HuggingFace rows loaded: {len(rows)}")

    inputs  = []
    skipped = 0

    for row in rows:
        code = row.get("code", "")
        if not code:
            skipped += 1
            continue

        test_cases = row.get("test_cases", [])
        stdin    = test_cases[0].get("input", "")  if test_cases else ""
        expected = test_cases[0].get("output", "") if test_cases else ""
        generator = row.get("test_case_generator", "")

        task_id = row.get("task_id", "")
        state   = state_map.get(task_id) or _build_default_state(row)

        inputs.append({
            "task_id":              task_id,
            "code":                 code,
            "stdin":                stdin,
            "expected":             expected,
            "test_case_generator":  generator,
            "prompt":               row.get("prompt", ""),
            "stress_level":         row.get("stress_level", "low"),
            "state":                {k: state.get(k, 0) for k in FEATURE_COLS},
        })

    return inputs, skipped


def build_from_pairs() -> list:
    state_map = _load_state_map()

    with open(PATHS["code_pairs"]) as f:
        first = json.loads(f.readline())
    print("code_pairs keys:", list(first.keys()))

    with open(PATHS["code_pairs"]) as f:
        records = [json.loads(line) for line in f if line.strip()]

    inputs  = []
    skipped = 0

    for rec in records:
        code = rec.get("refined_code", "") or rec.get("generated_code", "")
        if not code:
            skipped += 1
            continue

        task_id = rec.get("task_id", "")
        state   = state_map.get(task_id, {})
        if not state:
            skipped += 1
            continue

        test_cases = rec.get("test_cases", [])
        stdin    = rec.get("stress_input", "") or (test_cases[0].get("input", "")  if test_cases else "")
        expected = rec.get("ref_output", "")   or (test_cases[0].get("output", "") if test_cases else "")

        inputs.append({
            "task_id":             task_id,
            "code":                code,
            "description":         rec.get("prompt", ""),
            "stdin":               stdin,
            "expected":            expected,
            "test_case_generator": "",
            "stress_level":        "unknown",
            "source":              "effibench",
            "state":               {k: state.get(k, 0) for k in FEATURE_COLS},
        })

    return inputs, skipped


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",      choices=["hf", "pairs"], default="pairs",
                        help="hf=HuggingFace EffiBench, pairs=collected code_pairs file")
    parser.add_argument("--heavy-only",  action="store_true",
                        help="with --source hf: skip low-stress problems")
    args = parser.parse_args()

    if args.source == "hf":
        inputs, skipped = build_from_hf(heavy_only=args.heavy_only)
    else:
        inputs, skipped = build_from_pairs()

    out_path = PATHS["inputs"]
    with open(out_path, "w") as f:
        json.dump(inputs, f, indent=2)

    has_stdin = sum(1 for i in inputs if i.get("stdin", "").strip())
    has_gen   = sum(1 for i in inputs if i.get("test_case_generator", "").strip())
    total     = len(inputs) + skipped

    print(f"total={total}  skipped={skipped}  saved={len(inputs)}")
    print(f"  with stdin:     {has_stdin}/{len(inputs)}")
    print(f"  with generator: {has_gen}/{len(inputs)}")
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
