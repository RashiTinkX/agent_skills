"""
TEMPORARY (TEMP-CODEX): answer the engine's LLM calls with the local Codex
CLI instead of OpenRouter. For testing only -- remove once the agent-as-model
path (--emit-prompt / --facet-samples) replaces it. Enabled by
``generate_search_strategy.py --provider codex``.

How it works: facets.py and openrouter_client.py both send an OpenAI-style
chat request with ``requests.post(OPENROUTER_URL, json=payload)`` and read
back ``status_code`` / ``text`` / ``json()``. ``install()`` wraps
``requests.post`` so that requests to OpenRouter (and only those -- PubMed and
everything else pass through untouched) are answered by ``codex exec``, and
the answer is returned in the same chat-completions shape. No engine file is
changed, so parsing, voting, MeSH closure and the query builder are exactly
the code that runs with OpenRouter.

Differences from OpenRouter, worth stating in any result:
  * the model is Codex's (``CODEX_MODEL``, else Codex's default), not
    ``OPENROUTER_MODEL``;
  * ``temperature`` / ``seed`` / ``max_tokens`` from the payload are not
    applied -- Codex samples with its own settings.

Environment:
  CODEX_BIN    path to codex(.exe); default: PATH, then the VS Code extension
  CODEX_MODEL  model for ``codex exec -m``; default: Codex's configured model
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

OPENROUTER_PREFIX = "https://openrouter.ai/"
TIMEOUT = int(os.environ.get("CODEX_TIMEOUT", "300"))

_INSTRUCTIONS = (
    "You are being used as a plain chat-completion model, not as a coding agent. "
    "Do not run commands, read files or use tools. Follow the SYSTEM instructions "
    "below and answer the USER message exactly as that model would: reply with the "
    "JSON object only, no prose and no code fences."
)


def find_codex() -> str:
    explicit = os.environ.get("CODEX_BIN", "")
    if explicit:
        return explicit
    found = shutil.which("codex")
    if found:
        return found
    pattern = str(Path.home() / ".vscode" / "extensions" / "openai.chatgpt-*"
                  / "bin" / "*" / ("codex.exe" if os.name == "nt" else "codex"))
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]  # newest extension version
    raise RuntimeError("Codex CLI not found: install it or set CODEX_BIN")


class _Response:
    """The slice of requests.Response the engine reads."""

    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self.text = json.dumps(payload)
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _prompt(messages: list[dict]) -> str:
    parts = [_INSTRUCTIONS]
    for m in messages:
        parts.append(f"=== {str(m.get('role', 'user')).upper()} ===\n{m.get('content', '')}")
    return "\n\n".join(parts)


def complete(payload: dict) -> _Response:
    codex = find_codex()
    model = os.environ.get("CODEX_MODEL", "")
    with tempfile.TemporaryDirectory(prefix="codex_shim_") as work:
        out = Path(work) / "answer.txt"
        cmd = [codex, "exec", "--skip-git-repo-check", "--ephemeral",
               "-s", "read-only", "-C", work, "-o", str(out)]
        if model:
            cmd += ["-m", model]
        cmd.append("-")  # prompt on stdin: no command-line length limit
        try:
            proc = subprocess.run(cmd, input=_prompt(payload.get("messages", [])),
                                  capture_output=True, text=True, encoding="utf-8",
                                  timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return _Response(504, {"error": f"codex exec timed out after {TIMEOUT}s"})
        if proc.returncode != 0 or not out.exists():
            return _Response(502, {"error": f"codex exec exit {proc.returncode}: "
                                            f"{(proc.stderr or proc.stdout)[-400:]}"})
        content = out.read_text(encoding="utf-8").strip()
    return _Response(200, {"model": f"codex/{model or 'default'}",
                           "choices": [{"message": {"role": "assistant", "content": content}}]})


def install() -> str:
    """Route OpenRouter chat calls to Codex. Returns the codex binary in use."""
    import requests

    codex = find_codex()
    if getattr(requests.post, "_codex_shim", False):
        return codex
    original = requests.post

    def post(url, *args, **kwargs):
        if str(url).startswith(OPENROUTER_PREFIX):
            return complete(kwargs.get("json") or {})
        return original(url, *args, **kwargs)

    post._codex_shim = True
    requests.post = post
    # The engine skips the LLM when no key is set (closure falls back to its
    # heuristic). Codex needs none, so satisfy the check with a placeholder.
    os.environ.setdefault("OPENROUTER_API_KEY", "codex-shim")
    return codex
