import json
import os

_CACHE_PATH = "online_rl/cc_pool_cache.json"


def _build_pool_from_hf(min_rating: int, max_rating: int, pool_size: int) -> list:
    from datasets import load_dataset
    ds       = load_dataset("deepmind/code_contests", split="train", streaming=True)
    problems = []

    for row in ds:
        if len(problems) >= pool_size:
            break

        cf = row["cf_rating"]
        if not (min_rating <= cf <= max_rating):
            continue

        py3 = [s for l, s in zip(row["solutions"]["language"], row["solutions"]["solution"]) if l == 3]
        if not py3:
            continue

        gen_in  = row["generated_tests"]["input"]
        gen_out = row["generated_tests"]["output"]
        pub_in  = row["public_tests"]["input"]
        pub_out = row["public_tests"]["output"]

        if gen_in:
            idx      = max(range(len(gen_in)), key=lambda i: len(gen_in[i]))
            stdin    = gen_in[idx]
            expected = gen_out[idx] if idx < len(gen_out) else ""
        elif pub_in:
            stdin    = pub_in[0]
            expected = pub_out[0] if pub_out else ""
        else:
            continue

        problems.append({
            "task_id":      f"cc_{row['name']}",
            "description":  row["description"],
            "ref_solution": py3[0],
            "stdin":        stdin,
            "expected":     expected,
            "cf_rating":    cf,
            "cf_tags":      row["cf_tags"],
        })
        print(f"[loader] {row['name']} | cf={cf} | stdin={len(stdin)}chars")

    return problems


def load_cc_problems() -> list:
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH) as f:
            cache = json.load(f)
        cached = cache.get("problems", [])
        if cached:
            print(f"[loader] loaded {len(cached)} problems from cache ({_CACHE_PATH})")
            return cached

    print(f"[loader] streaming from HuggingFace (first time only)...")
    problems = _build_pool_from_hf(min_rating, max_rating, pool_size)

    with open(_CACHE_PATH, "w") as f:
        json.dump({"meta": {"min_rating": min_rating, "max_rating": max_rating}, "problems": problems}, f)
    print(f"[loader] cached {len(problems)} problems to {_CACHE_PATH}")

    return problems
