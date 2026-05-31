"""
llm.py — Code generation via Laguna M.1 (OpenRouter) for normal problems,
         Claude claude-sonnet-4-5 (Anthropic) for high-stress problems.
"""

import ast
import os
import re
import sys
from pathlib import Path

import anthropic
import requests

# ── Load .env ─────────────────────────────────────────────────────────────────
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Config ────────────────────────────────────────────────────────────────────

MINIMAX_MODEL   = "poolside/laguna-m.1:free"
CLAUDE_MODEL    = "claude-sonnet-4-5"
HAIKU_MODEL     = "claude-haiku-4-5-20251001"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_HEADERS = {
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/madiha/microforge-rl",
    "X-Title":      "microforge-rl thesis",
}


def generate_code(prompt: str, prefill: str = "") -> str:
    """Call Laguna M.1 via OpenRouter and return the generated code string."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    headers = {**_HEADERS, "Authorization": f"Bearer {api_key}"}

    messages = [{"role": "user", "content": prompt}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    response = requests.post(
        url=_OPENROUTER_URL,
        headers=headers,
        json={
            "model":       MINIMAX_MODEL,
            "max_tokens":  8192,
            "temperature": 0.2,
            "messages":    messages,
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise Exception(f"OpenRouter error {response.status_code}: {response.text}")

    code = response.json()["choices"][0]["message"]["content"] or ""
    if prefill:
        code = prefill + code
    return strip_code(code)


def generate_code_claude(prompt: str, model: str = CLAUDE_MODEL) -> str:
    """Call Claude via Anthropic SDK. Uses Sonnet by default, pass HAIKU_MODEL for cheaper calls."""
    api_key = os.environ.get("ClAUDE_CONSOLE_API_KEY", "")
    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text if message.content else ""
    return strip_code(text)


def warmup_test() -> None:
    """Run a single test call to verify the API key and model are working."""
    print("[llm] Running Laguna warmup test...")
    try:
        result = generate_code("Write a Python script that prints the sum of 1 to 100")
        print("[llm] Laguna test output:")
        print(result)
        print("[llm] Warmup OK\n")
    except Exception as e:
        print(f"[llm] FATAL: Laguna warmup failed: {e}", file=sys.stderr)
        sys.exit(1)


def call_minimax(prompt: str, expected_func: str = None, retries: int = 3,
                 prefill: str = "", stress_level: str = "low") -> tuple[str, int, int]:
    """
    Generate code. Routes to Claude for high-stress problems, Laguna otherwise.
    Returns (generated_text, prompt_tokens, completion_tokens).
    """
    if stress_level in ("high", "medium"):
        model_name = CLAUDE_MODEL
    else:
        model_name = HAIKU_MODEL
    use_claude = True
    print(f"[llm] Using {model_name} | stress={stress_level}")

    text = ""
    for attempt in range(retries + 1):
        try:
            if stress_level in ("high", "medium"):
                text = generate_code_claude(prompt, model=CLAUDE_MODEL)
            else:
                text = generate_code_claude(prompt, model=HAIKU_MODEL)
        except Exception as e:
            print(f"[llm] WARNING: attempt {attempt+1} API error: {e}, retrying...")
            continue

        prompt_tokens     = len(prompt.split())
        completion_tokens = len(text.split())
        print(f"[llm] attempt {attempt+1}: {len(text)} chars")

        if not text or len(text) < 10:
            print(f"[llm] WARNING: attempt {attempt+1} returned empty code, retrying...")
            continue

        try:
            ast.parse(text)
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False

        func_ok = not expected_func or expected_func in text

        if syntax_ok and func_ok:
            return text, prompt_tokens, completion_tokens

        reasons = []
        if not func_ok:
            reasons.append(f"missing function `{expected_func}`")
        if not syntax_ok:
            reasons.append("syntax error")
        print(f"[llm] WARNING: attempt {attempt+1} failed ({', '.join(reasons)}), retrying...")

    return text, len(prompt.split()), len(text.split())


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    text = text.replace('&nbsp;', ' ').replace('&le;', '<=').replace('&ge;', '>=')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


_CODE_ONLY_HEADER = (
    "IMPORTANT: Your response must contain ONLY valid Python code. "
    "No explanation, no prose, no markdown fences. "
    "If you write anything other than Python code, it will cause a syntax error.\n\n"
)


def build_llm_prompt(instance: dict) -> tuple[str, str | None, str]:
    """Returns (prompt_text, expected_func_name, prefill)."""
    func_name = None
    if instance.get("test_list"):
        m = re.match(r'assert (\w+)\(', instance["test_list"][0])
        if m:
            func_name = m.group(1)
    elif instance.get("entry_point"):
        func_name = instance["entry_point"]

    dataset   = instance.get("dataset", "")
    task_text = _strip_html(instance["prompt"])

    if dataset in ("effibench", "effibench_large", "security"):
        func_hint = f" Name the main function `{func_name}`." if func_name else ""
        prompt = (
            _CODE_ONLY_HEADER
            + "Write a complete Python script that reads input from stdin and writes the answer to stdout.\n\n"
            + task_text
            + func_hint
        )
    else:
        func_hint = f" Name the function `{func_name}`." if func_name else ""
        prompt = (
            _CODE_ONLY_HEADER
            + "Write a Python function that solves the following task.\n\n"
            + task_text
            + func_hint
        )

    return prompt, func_name, ""


def strip_code(text: str) -> str:
    """Strip markdown fences if present, otherwise return as-is."""
    fenced = re.search(r'```(?:python)?\n?(.*?)```', text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()
