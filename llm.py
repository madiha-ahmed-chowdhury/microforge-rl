"""
llm.py — MiniMax API interface.

All LLM calls go through here. Import this module to generate code from prompts.
"""

import os
import re
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

# ── Config ────────────────────────────────────────────────────────────────────

MINIMAX_MODEL    = "MiniMax-M2.7"
MINIMAX_BASE_URL = "https://api.minimax.io/anthropic"


def call_minimax(prompt: str) -> tuple[str, int, int]:
    """
    Call the MiniMax API with a prompt.
    Returns (generated_text, prompt_tokens, completion_tokens).
    """
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=MINIMAX_BASE_URL,
    )
    message = client.messages.create(
        model=MINIMAX_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = ""
    for block in message.content:
        if block.type == "text":
            text = block.text
            break
    return text, message.usage.input_tokens, message.usage.output_tokens


def build_llm_prompt(instance: dict) -> str:
    """
    Build the prompt to send to the LLM for a dataset instance.
    Extracts the expected function name from test cases and appends it
    so the model uses the correct name.
    """
    func_name = None
    if instance.get("test_list"):
        m = re.match(r'assert (\w+)\(', instance["test_list"][0])
        if m:
            func_name = m.group(1)
    elif instance.get("entry_point"):
        func_name = instance["entry_point"]
    hint = f" Name the function `{func_name}`." if func_name else ""
    return (
        instance["prompt"]
        + hint
        + "\n\nReturn only the Python code, no explanation, no markdown."
    )


def strip_code(text: str) -> str:
    """Strip markdown fences and prose from LLM output, leaving only code."""
    fenced = re.search(r'```(?:python)?\n?(.*?)```', text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    clean = []
    for line in text.splitlines():
        s = line.strip()
        is_prose = (
            s and s[0].isupper()
            and '=' not in s and '(' not in s
            and ':' not in s and not s.startswith('#')
        )
        if not is_prose:
            clean.append(line)
    return '\n'.join(clean).strip()
