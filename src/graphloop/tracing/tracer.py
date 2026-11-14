"""Structured execution traces shared by every agent.

Traces are the single source of truth for cost metrics (tool calls, LLM calls, tokens) and
power the execution overlay in the HTML report.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class EventKind(StrEnum):
    NODE = "node"
    TOOL = "tool"
    LLM = "llm"
    BELIEF = "belief"
    DIAGNOSIS = "diagnosis"
    NOTE = "note"
    DEGRADED = "degraded"  # a cap or outage forced a loud, rule-based fallback
    GOVERNANCE = "governance"  # proposals, approvals, executions, read-backs


@dataclass(frozen=True, slots=True)
class TraceEvent:
    seq: int
    kind: EventKind
    name: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceSummary:
    tool_calls: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    invalid_llm_outputs: int
    nodes_visited: int
    cost_usd: float
    degraded: bool

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Tracer:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(self, kind: EventKind, name: str, **data: Any) -> TraceEvent:
        event = TraceEvent(len(self.events), kind, name, data)
        self.events.append(event)
        return event

    def of(self, kind: EventKind) -> list[TraceEvent]:
        return [e for e in self.events if e.kind is kind]

    def summary(self) -> TraceSummary:
        llm = self.of(EventKind.LLM)
        return TraceSummary(
            tool_calls=len(self.of(EventKind.TOOL)),
            llm_calls=len(llm),
            prompt_tokens=sum(int(e.data["prompt_tokens"]) for e in llm),
            completion_tokens=sum(int(e.data["completion_tokens"]) for e in llm),
            invalid_llm_outputs=sum(1 for e in llm if not e.data["valid"]),
            nodes_visited=len(self.of(EventKind.NODE)),
            cost_usd=sum(float(e.data.get("cost_usd", 0.0)) for e in llm),
            degraded=bool(self.of(EventKind.DEGRADED)),
        )

    def to_jsonl(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for e in self.events:
                fh.write(json.dumps({**asdict(e), "kind": e.kind.value}, default=str) + "\n")
