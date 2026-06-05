"""
llm_cc.py — Code generation for CodeContests problems.

Uses updated Claude models with a three-tier escalation:
  Haiku 4.5   -> first attempt (syntax check only)
  Sonnet 4.6  -> default generation for hard problems
  Opus 4.8    -> refinement escalation when Sonnet fails
"""

import ast
import os
import re
import sys
from pathlib import Path

import anthropic

# ── Load .env ─────────────────────────────────────────────────────────────────
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Models ────────────────────────────────────────────────────────────────────

HAIKU_MODEL  = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"
OPUS_MODEL   = "claude-opus-4-8"

# Escalation order: attempt 0 -> Sonnet, attempt 1+ -> Opus
REFINEMENT_MODELS = [SONNET_MODEL, OPUS_MODEL]

_CODE_ONLY_HEADER = (
    "IMPORTANT: Your response must contain ONLY valid Python code. "
    "No explanation, no prose, no markdown fences. "
    "If you write anything other than Python code, it will cause a syntax error.\n\n"
)


# ── Core generation ───────────────────────────────────────────────────────────

def generate_code_claude(prompt: str, model: str = SONNET_MODEL) -> str:
    api_key = os.environ.get("ClAUDE_CONSOLE_API_KEY", "")
    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text if message.content else ""
    return strip_code(text)


def generate_cc_code(prompt: str, retries: int = 2) -> tuple[str, int, int]:
    """
    Generate code for a CodeContests problem.
    First attempt uses Sonnet, retries use Opus.
    Returns (generated_text, prompt_tokens, completion_tokens).
    """
    models = [SONNET_MODEL] + [OPUS_MODEL] * retries
    text   = ""

    for attempt, model in enumerate(models):
        print(f"[llm_cc] attempt {attempt + 1} using {model}")
        try:
            text = generate_code_claude(prompt, model=model)
        except Exception as e:
            print(f"[llm_cc] WARNING: attempt {attempt + 1} API error: {e}, retrying...")
            continue

        if not text or len(text) < 10:
            print(f"[llm_cc] WARNING: attempt {attempt + 1} returned empty code")
            continue

        try:
            ast.parse(text)
            print(f"[llm_cc] attempt {attempt + 1}: {len(text)} chars, syntax OK")
            return text, len(prompt.split()), len(text.split())
        except SyntaxError:
            print(f"[llm_cc] WARNING: attempt {attempt + 1} syntax error, escalating...")

    return text, len(prompt.split()), len(text.split())


def refine_cc_code(original_code: str, stdin: str, expected: str,
                   actual: str, problem: str, attempt: int = 0) -> str:
    """
    Refine wrong code. Escalates from Sonnet to Opus.
    attempt=0 -> Sonnet, attempt=1+ -> Opus
    """
    model  = REFINEMENT_MODELS[min(attempt, len(REFINEMENT_MODELS) - 1)]
    prompt = (
        "The following Python code produced wrong output on a competitive programming problem.\n"
        f"Problem:\n{problem[:1000]}\n\n"
        f"Code:\n{original_code}\n\n"
        f"Input used:\n{stdin[:500]}\n\n"
        f"Expected output:\n{expected[:300]}\n\n"
        f"Actual output:\n{actual[:300]}\n\n"
        "Return only the corrected Python code. "
        "Read from stdin and write to stdout."
    )
    print(f"[llm_cc] refinement attempt {attempt} using {model}")
    try:
        return generate_code_claude(prompt, model=model)
    except Exception as e:
        print(f"[llm_cc] refinement error: {e}")
        return ""


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_cc_prompt(description: str, sample_input: str = "", sample_output: str = "") -> str:
    prompt = (
        _CODE_ONLY_HEADER
        + "Write a complete Python script that reads input from stdin and writes the answer to stdout.\n\n"
        + description.strip()
    )
    if sample_input:
        prompt += f"\n\nExample input:\n{sample_input[:400]}"
    if sample_output:
        prompt += f"\nExample output:\n{sample_output[:200]}"
    return prompt


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_code(text: str) -> str:
    fenced = re.search(r'```(?:python)?\n?(.*?)```', text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()
