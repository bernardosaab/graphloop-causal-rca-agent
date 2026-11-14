"""Prompt rendering shared by every client, so token accounting is comparable across them."""

from __future__ import annotations

import json

from graphloop.llm.base import DecisionRequest

SYSTEM_PROMPT = (
    "You are an on-call site reliability engineer investigating a production incident. "
    "Reason only from the facts provided. Choose exactly one option by its id. "
    'Answer with JSON: {"option_id": "<id>", "rationale": "<one sentence>"}.'
)


def render_prompt(request: DecisionRequest) -> str:
    facts = "\n".join(f"- {f.text}" for f in request.facts) or "- (none yet)"
    options = "\n".join(f"- [{o.id}] {o.text}" for o in request.options)
    return (
        f"# Task: {request.purpose}\n{request.instructions}\n\n"
        f"## Facts\n{facts}\n\n## Options\n{options}\n"
    )


def response_schema(request: DecisionRequest) -> dict[str, object]:
    """JSON schema constraining the answer to the option ids (used by Ollama's ``format``)."""
    return {
        "type": "object",
        "properties": {
            "option_id": {"type": "string", "enum": [o.id for o in request.options]},
            "rationale": {"type": "string"},
        },
        "required": ["option_id", "rationale"],
    }


def estimate_tokens(text: str) -> int:
    """Cheap, model-agnostic estimate (~4 characters per token)."""
    return max(1, len(text) // 4)


def render_answer(option_id: str, rationale: str) -> str:
    return json.dumps({"option_id": option_id, "rationale": rationale})
