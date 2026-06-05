import ast
import os

from online_rl.config import MAX_REFINEMENT_ATTEMPTS


def _try_parse(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _build_gen_prompt(description: str, stdin: str, expected: str) -> str:
    prompt = (
        "IMPORTANT: Your response must contain ONLY valid Python code. "
        "No explanation, no prose, no markdown fences.\n\n"
        "Write a complete Python script that reads from stdin and writes to stdout.\n\n"
        + description[:1500]
    )
    if stdin:
        prompt += f"\n\nExample input:\n{stdin[:300]}"
    if expected:
        prompt += f"\nExample output:\n{expected[:200]}"
    return prompt


def _try_laguna(prompt: str) -> str | None:
    import requests
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    try:
        resp = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer":  "https://github.com/madiha/microforge-rl",
            },
            json={
                "model":       "poolside/laguna-m.1:free",
                "max_tokens":  8192,
                "temperature": 0.2,
                "messages":    [{"role": "user", "content": prompt}],
            },
            timeout=180,
        )
        if resp.status_code == 200:
            data    = resp.json()["choices"][0]
            content = data["message"]["content"]
            finish  = data["finish_reason"]
            if content and finish == "stop":
                from llm_cc import strip_code
                code = strip_code(content)
                if _try_parse(code):
                    return code
        print(f"[llm_caller] Laguna failed (status={resp.status_code})")
    except Exception as e:
        print(f"[llm_caller] Laguna error: {e}")
    return None


def _try_claude(prompt: str, model: str) -> str | None:
    from llm_cc import generate_code_claude
    try:
        code = generate_code_claude(prompt, model=model)
        if _try_parse(code):
            return code
        print(f"[llm_caller] {model} syntax error")
    except Exception as e:
        print(f"[llm_caller] {model} error: {e}")
    return None


def generate_code(description: str, stdin: str, expected: str,
                  tier: str, fallback_code: str) -> tuple[str, str]:
    from llm_cc import SONNET_MODEL, OPUS_MODEL
    prompt = _build_gen_prompt(description, stdin, expected)

    if tier == "laguna":
        code = _try_laguna(prompt)
        if code:
            return code, "laguna-m.1"
        print("[llm_caller] Laguna failed — escalating to Sonnet")
        code = _try_claude(prompt, SONNET_MODEL)
        if code:
            return code, "sonnet-escalated"
        print("[llm_caller] Sonnet failed — escalating to Opus")
        code = _try_claude(prompt, OPUS_MODEL)
        if code:
            return code, "opus-escalated"

    elif tier == "sonnet":
        code = _try_claude(prompt, SONNET_MODEL)
        if code:
            return code, SONNET_MODEL
        print("[llm_caller] Sonnet failed — escalating to Opus")
        code = _try_claude(prompt, OPUS_MODEL)
        if code:
            return code, "opus-escalated"

    elif tier == "opus":
        code = _try_claude(prompt, OPUS_MODEL)
        if code:
            return code, OPUS_MODEL

    print("[llm_caller] all LLMs failed — using ref solution")
    return fallback_code, "ref-fallback"


def refine_code(original_code: str, stdin: str, expected: str,
                actual: str, description: str, attempt: int = 0) -> str:
    from llm_cc import generate_code_claude, SONNET_MODEL, OPUS_MODEL
    models = [SONNET_MODEL, OPUS_MODEL]
    model  = models[min(attempt, len(models) - 1)]
    prompt = (
        "The following code produced wrong output on a competitive programming problem.\n"
        f"Problem:\n{description[:800]}\n\n"
        f"Code:\n{original_code}\n\n"
        f"Input:\n{stdin[:500]}\n\n"
        f"Expected:\n{expected[:300]}\n\n"
        f"Actual:\n{actual[:300]}\n\n"
        "Return only the corrected Python code."
    )
    print(f"[llm_caller] refinement attempt {attempt} using {model}")
    try:
        return generate_code_claude(prompt, model=model)
    except Exception as e:
        print(f"[llm_caller] refinement error: {e}")
        return ""


def run_prep_vm(vm, code: str, stdin: str, expected: str,
                description: str, run_code_fn) -> tuple[str, bool]:
    """
    Run LLM code on PREP VM, refine if wrong.
    Returns (final_code, passed).
    """
    result     = run_code_fn(vm, code, stdin, 30_000)
    actual_out = result.get("stdout", "").strip()
    passed     = (
        result.get("exit_code") == 0 and
        bool(expected) and
        actual_out == expected.strip()
    )

    if not passed and result.get("exit_code") == 0 and expected:
        print("[llm_caller] LLM code wrong — refining...")
        for attempt in range(MAX_REFINEMENT_ATTEMPTS):
            refined = refine_code(code, stdin, expected, actual_out, description, attempt)
            if not refined:
                break
            code    = refined
            retry   = run_code_fn(vm, code, stdin, 30_000)
            actual_out = retry.get("stdout", "").strip()
            if retry.get("exit_code") == 0 and actual_out == expected.strip():
                print(f"[llm_caller] refinement {attempt+1} passed ✓")
                passed = True
                break
            print(f"[llm_caller] refinement {attempt+1} still wrong")

    return code, passed
