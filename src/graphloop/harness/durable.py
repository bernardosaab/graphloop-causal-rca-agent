"""Durable execution: checkpoint every step, resume after transient failures, give up loudly.

LangGraph provides the mechanism (a checkpointer that saves state after each node); the
policy lives here: resume from the last completed step at most ``max_resumes`` times, then
mark the run interrupted instead of retrying forever.
"""

from __future__ import annotations

from typing import Any, Protocol

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from graphloop.simulator.environment import TransientToolError
from graphloop.tracing.tracer import EventKind, Tracer

DEFAULT_MAX_RESUMES = 2


class RunInterruptedError(RuntimeError):
    """The run failed more times than the resume policy allows."""


class _Invocable(Protocol):
    def invoke(self, input: Any, config: Any) -> Any: ...


#: Explicit allow-list of types that may be revived from a checkpoint (deny by default).
CHECKPOINT_TYPES: tuple[tuple[str, str], ...] = (
    ("graphloop.domain.vocabulary", "Diagnosis"),
    ("graphloop.domain.vocabulary", "FaultType"),
    ("graphloop.domain.vocabulary", "Finding"),
    ("graphloop.domain.vocabulary", "Signal"),
    ("graphloop.domain.vocabulary", "RemediationAction"),
    ("graphloop.domain.belief", "BeliefState"),
    ("graphloop.llm.base", "Fact"),
    ("graphloop.governance.proposals", "ChangeProposal"),
    ("graphloop.governance.proposals", "RiskTier"),
    ("graphloop.governance.policy", "Approval"),
)


def new_checkpointer() -> InMemorySaver:
    """In-memory for tests and benchmarks; swap for a SQLite/Postgres saver in production."""
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))


def invoke_durably(
    app: _Invocable,
    inputs: dict[str, Any],
    thread_id: str,
    tracer: Tracer,
    recursion_limit: int = 200,
    max_resumes: int = DEFAULT_MAX_RESUMES,
) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    payload: dict[str, Any] | None = inputs
    resumes = 0
    while True:
        try:
            result: dict[str, Any] = app.invoke(payload, config)
            return result
        except TransientToolError as exc:
            if resumes >= max_resumes:
                tracer.emit(
                    EventKind.NOTE,
                    "interrupted",
                    thread_id=thread_id,
                    error=str(exc),
                    resumes=resumes,
                )
                raise RunInterruptedError(f"{thread_id}: gave up after {resumes} resumes") from exc
            resumes += 1
            tracer.emit(
                EventKind.NOTE,
                "resumed_from_checkpoint",
                thread_id=thread_id,
                attempt=resumes,
                error=str(exc),
            )
            payload = None  # None => continue from the last saved checkpoint
