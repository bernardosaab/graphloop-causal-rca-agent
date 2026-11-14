"""Adapter for a local Ollama server (https://ollama.com): free, offline, open-weight models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from graphloop.llm.base import Decision, DecisionRequest, TokenUsage
from graphloop.llm.prompt import SYSTEM_PROMPT, estimate_tokens, render_prompt, response_schema


class LLMUnavailableError(RuntimeError):
    pass


@dataclass
class OllamaClient:
    model: str = "qwen2.5:7b-instruct"
    # Same variable the Ollama CLI uses, so containers can point at an `ollama` service.
    base_url: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )
    timeout_s: float = 120.0
    max_retries: int = 1
    transport: httpx.BaseTransport | None = None  # injectable for tests
    _http: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if "://" not in self.base_url:  # Ollama accepts "host:port" without a scheme
            self.base_url = f"http://{self.base_url}"
        self._http = httpx.Client(
            base_url=self.base_url, timeout=self.timeout_s, transport=self.transport
        )

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def decide(self, request: DecisionRequest) -> Decision:
        prompt = render_prompt(request)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        prompt_tokens = completion_tokens = 0
        for _ in range(self.max_retries + 1):
            body = self._chat(messages, response_schema(request))
            prompt_tokens += int(body.get("prompt_eval_count", estimate_tokens(prompt)))
            completion_tokens += int(body.get("eval_count", 0))
            message = body.get("message", {})
            content = str(message.get("content", "")) if isinstance(message, dict) else ""
            parsed = _parse(content, {o.id for o in request.options})
            if parsed is not None:
                return Decision(parsed[0], parsed[1], TokenUsage(prompt_tokens, completion_tokens))
            messages += [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": "Invalid answer. Reply with one option_id from the list.",
                },
            ]
        # Degrade gracefully: take the first option and flag the decision as invalid.
        return Decision(
            request.options[0].id,
            "invalid model output",
            TokenUsage(prompt_tokens, completion_tokens),
            valid=False,
        )

    def _chat(self, messages: list[dict[str, str]], schema: dict[str, object]) -> dict[str, Any]:
        try:
            resp = self._http.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "format": schema,
                    "stream": False,
                    "options": {"temperature": 0, "seed": 0},
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Ollama request failed: {exc}") from exc
        data: dict[str, Any] = resp.json()
        return data


def _parse(content: str, valid_ids: set[str]) -> tuple[str, str] | None:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    option_id = data.get("option_id")
    if option_id not in valid_ids:
        return None
    return str(option_id), str(data.get("rationale", ""))
