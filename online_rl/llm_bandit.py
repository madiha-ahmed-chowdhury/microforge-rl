import json
import os
import random

MODEL_OPTIONS = [
    {
        "name":      "gpt-oss-120b",
        "model_str": "openai/gpt-oss-120b:free",
        "key_env":   "OPENROUTER_API_KEY",
        "cost":      0.0,
        "tier":      "free",
    },
    {
        "name":      "kimi-k2",
        "model_str": "moonshotai/kimi-k2.6:free",
        "key_env":   "OPENROUTER_API_KEY_2",
        "cost":      0.0,
        "tier":      "free",
    },
    {
        "name":      "qwen-coder",
        "model_str": "qwen/qwen3-coder:free",
        "key_env":   "OPENROUTER_API_KEY_3",
        "cost":      0.0,
        "tier":      "free",
    },
    {
        "name":      "claude-sonnet",
        "model_str": "claude-sonnet-4-5",
        "key_env":   "ANTHROPIC_API_KEY",
        "cost":      0.3,
        "tier":      "sonnet",
    },
    {
        "name":      "claude-opus",
        "model_str": "claude-opus-4-5",
        "key_env":   "ANTHROPIC_API_KEY",
        "cost":      1.0,
        "tier":      "opus",
    },
]

_N_BUCKETS    = 3
_N_MODELS     = len(MODEL_OPTIONS)
_BUCKET_NAMES = ["easy", "medium", "hard"]


def count_examples(description: str) -> int:
    count = description.lower().count("output")
    return max(0, min(count - 1, 5))


class LLMBandit:
    """
    Thompson Sampling contextual bandit for LLM model selection.
    3 context buckets (easy/medium/hard) × 5 model options.
    Bucketing uses only problem-level signals: cf_rating and example count.
    No code features — code does not exist yet when the bandit decides.
    """

    def __init__(self):
        self.alpha = [[1.0] * _N_MODELS for _ in range(_N_BUCKETS)]
        self.beta  = [[1.0] * _N_MODELS for _ in range(_N_BUCKETS)]

    def get_bucket(self, problem: dict) -> int:
        rating   = int(problem.get("cf_rating", 1200) or 1200)
        desc     = problem.get("description", "")
        desc_len = len(desc)
        examples = count_examples(desc)

        if rating <= 1200:
            bucket = 0
        elif rating <= 1600:
            bucket = 1
        else:
            bucket = 2

        if bucket < 2 and (desc_len > 1000 or examples >= 3):
            bucket += 1

        return bucket

    def select(self, problem: dict) -> tuple[dict, int]:
        bucket  = self.get_bucket(problem)
        samples = [
            random.betavariate(self.alpha[bucket][i], self.beta[bucket][i])
            for i in range(_N_MODELS)
        ]
        idx = samples.index(max(samples))
        return MODEL_OPTIONS[idx], idx

    def update(self, problem: dict, model_idx: int,
               passed, used_ref_fallback: bool) -> None:
        bucket = self.get_bucket(problem)

        r_quality  = 1.0 if passed else -1.0
        r_cost     = -MODEL_OPTIONS[model_idx]["cost"]
        r_fallback = -2.0 if used_ref_fallback else 0.0
        total = r_quality + r_cost + r_fallback

        if total > 0:
            self.alpha[bucket][model_idx] += total
        else:
            self.beta[bucket][model_idx]  += abs(total)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"alpha": self.alpha, "beta": self.beta}, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.alpha = data["alpha"]
        self.beta  = data["beta"]

    def summary(self) -> None:
        print("[bandit] Thompson Sampling summary (mean win-prob per bucket/model):")
        for b, bname in enumerate(_BUCKET_NAMES):
            parts = []
            for i, m in enumerate(MODEL_OPTIONS):
                a, be = self.alpha[b][i], self.beta[b][i]
                mean  = a / (a + be)
                n_upd = round(a + be - 2.0, 1)
                parts.append(f"{m['name']}={mean:.3f}(n={n_upd})")
            print(f"  {bname:6s}: {' | '.join(parts)}")
