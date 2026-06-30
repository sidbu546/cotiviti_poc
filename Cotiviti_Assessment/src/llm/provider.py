"""
LLM provider abstraction — Ollama only.
Wraps chat and structured-output calls with a consistent interface.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import ollama
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def _client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


def chat(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    """Single-turn chat. Returns the assistant message content."""
    model = model or DEFAULT_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _client().chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    return response["message"]["content"]


def chat_json(
    prompt: str,
    system: str = "",
    model: str | None = None,
) -> dict[str, Any]:
    """
    Single-turn chat with forced JSON output via Ollama's format parameter.
    Returns a parsed dict. Raises ValueError if JSON cannot be parsed.
    """
    model = model or DEFAULT_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _client().chat(
        model=model,
        messages=messages,
        format="json",
        options={"temperature": 0.0},
    )
    content = response["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw output:\n{content}") from e
