"""The LLM port.

Every model interaction in this project is a *constrained decision*: given facts and a
finite list of options, pick one. That shape works for real models (rendered as a prompt
and answered as JSON) and for the deterministic :class:`SimulatedLLM`, and it makes
hallucinated actions detectable instead of silently executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Fact:
    """A piece of context.

    ``refs`` are the entities it mentions, ``salience`` how loud it is, and ``exonerates`` the
    entities it argues *against* (e.g. "A timed out calling B" exonerates A and implicates B).
    """

    text: str
    refs: frozenset[str]
    salience: float
    exonerates: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class Option:
    """A candidate action.

    ``subject`` is the primary entity the option acts on (e.g. a service); ``refs`` are all
    entities it mentions. ``commits`` marks options that end the investigation.
    """

    id: str
    text: str
    subject: str
    refs: frozenset[str]
    commits: bool = False


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    purpose: str
    instructions: str
    facts: tuple[Fact, ...]
    options: tuple[Option, ...]

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("a decision needs at least one option")
        ids = [o.id for o in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError("option ids must be unique")

    def option(self, option_id: str) -> Option:
        for o in self.options:
            if o.id == option_id:
                return o
        raise KeyError(option_id)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class Decision:
    option_id: str
    rationale: str
    usage: TokenUsage
    valid: bool = True  # False when the model answered outside the option set


class LLMClient(Protocol):
    @property
    def name(self) -> str: ...

    def decide(self, request: DecisionRequest) -> Decision: ...
