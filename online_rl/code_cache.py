import json
import os
import time


class CodeCache:

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache = {}
        self.load()

    def has(self, task_id: str) -> bool:
        return task_id in self.cache

    def get(self, task_id: str) -> dict | None:
        return self.cache.get(task_id)

    def store(self, task_id: str, code: str,
              model_used: str, passed: bool,
              tests_passed) -> None:
        self.cache[task_id] = {
            "code":         code,
            "model_used":   model_used,
            "passed":       passed,
            "tests_passed": tests_passed,
            "cached_at":    time.time(),
            "use_count":    0,
        }
        self.save()

    def increment_use(self, task_id: str) -> None:
        if task_id in self.cache:
            self.cache[task_id]["use_count"] += 1
            if self.cache[task_id]["use_count"] % 50 == 0:
                self.save()

    def should_refresh(self, task_id: str,
                       max_failures: int = 5) -> bool:
        entry = self.cache.get(task_id)
        if not entry:
            return True
        # refresh if code was consistently wrong
        # but not if it was resource failures (those are SAC's job)
        if entry["tests_passed"] is False:
            return True
        return False

    def invalidate(self, task_id: str) -> None:
        if task_id in self.cache:
            del self.cache[task_id]
            self.save()

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def load(self) -> None:
        if os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                self.cache = json.load(f)
            print(f"[cache] loaded {len(self.cache)} cached problems")
        else:
            self.cache = {}

    def stats(self) -> str:
        total      = len(self.cache)
        passed     = sum(1 for v in self.cache.values() if v["passed"])
        use_counts = [v["use_count"] for v in self.cache.values()]
        avg_use    = sum(use_counts) / max(len(use_counts), 1)
        return (f"cached={total} passed={passed} "
                f"avg_use={avg_use:.1f}")
