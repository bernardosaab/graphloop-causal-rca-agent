"""LangGraph state for the graph agent, plus its tunable policy.

State holds only checkpoint-friendly values (strings, numbers, lists and frozen dataclasses),
so every step can be persisted and a crashed run resumed from its last completed node.
Probes are encoded as ``"<tool>:<service>"`` strings (see ``agents.common.probe_key``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from graphloop.domain.belief import BeliefState
from graphloop.domain.vocabulary import Diagnosis, Finding


@dataclass(frozen=True)
class GraphAgentConfig:
    confidence_threshold: float = 0.8  # stop verifying once a hypothesis is this likely
    max_verify_iterations: int = 10  # hard bound on the inner verification loop
    max_reflections: int = 2  # hard bound on the critique -> re-verify loop
    max_rejections: int = 2  # hard bound on counterfactual rejections
    top_k: int = 3  # hypotheses considered per decision
    memory_window: int | None = None  # only trust causal memory this recent (valid time)
    enable_recipes: bool = True  # allow the compiled, model-free path
    # Without a failure signature, still accept when the timeout trail ends at the hypothesis
    # and the posterior is at least this high. ``None`` demands a signature every time.
    chain_acceptance: float | None = 0.85


class GraphState(TypedDict, total=False):
    incident_id: str
    symptom: str
    belief: BeliefState
    findings: list[Finding]
    evidence_ids: list[str]
    probed: list[str]
    iterations: int
    reflections: int
    rejections: int
    forced_probe: str | None
    next_probe: str | None
    exhausted: bool
    recipe: str | None  # the compiled recipe that fired, if any
    diagnosis: Diagnosis | None
    confidence: float
    hallucinated: int
    verdict: str
