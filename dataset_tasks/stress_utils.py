from pathlib import Path


def load_stress_input(problem: dict, base_dir: Path) -> str:
    path = problem.get("stress_input_path")
    if not path:
        cases = problem.get("test_cases", [])
        return cases[0]["input"] if cases else ""
    full_path = base_dir / path
    if not full_path.exists():
        cases = problem.get("test_cases", [])
        return cases[0]["input"] if cases else ""
    return full_path.read_text(encoding="utf-8")
