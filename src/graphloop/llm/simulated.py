"""A deterministic, *agent-agnostic* stand-in for a language model.

It models a bounded-competence reasoner with three well-documented LLM failure modes:

1. **Salience-driven attention**: an option is scored by the visible facts that mention its
   subject, weighted by how alarming each fact looks and how many entities they share.
2. **Limited effective context** ("lost in the middle"): only the first ``pinned`` and the
   last ``context_window - pinned`` facts are attended to.
3. **Stochastic slips**: with probability ``slip_rate`` it picks one of the runner-up options.

It also needs ``commit_threshold`` worth of support before it ends an investigation while
other actions remain. It never sees which agent is calling it or what the ground truth is;
see ``docs/adr/0005-simulated-llm-evaluation.md`` for the fairness argument and its limits.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from graphloop.llm.base import Decision, DecisionRequest, Fact, Option, TokenUsage
from graphloop.llm.prompt import SYSTEM_PROMPT, estimate_tokens, render_answer, render_prompt


@dataclass
class SimulatedLLM:
    seed: int = 0
    slip_rate: float = 0.1
    context_window: int = 14
    pinned: int = 3
    commit_threshold: float = 3.0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.slip_rate <= 1.0:
            raise ValueError("slip_rate must be within [0, 1]")
        if self.context_window <= self.pinned:
            raise ValueError("context_window must exceed pinned")
        self._rng = random.Random(self.seed)

    @property
    def name(self) -> str:
        return f"simulated(slip={self.slip_rate}, window={self.context_window})"

    def decide(self, request: DecisionRequest) -> Decision:
        visible = self.attend(request.facts)
        scored = sorted(
            ((self.score(o, visible), i, o) for i, o in enumerate(request.options)),
            key=lambda t: (-t[0], t[1]),
        )
        eligible = [t for t in scored if self._may_choose(t[2], t[0], request)]
        choice = eligible[0]
        if len(eligible) > 1 and self._rng.random() < self.slip_rate:
            choice = self._rng.choice(eligible[1:3])
        score, _, option = choice
        rationale = f"strongest support ({score:.2f}) among visible facts"
        prompt = SYSTEM_PROMPT + render_prompt(request)
        usage = TokenUsage(
            estimate_tokens(prompt), estimate_tokens(render_answer(option.id, rationale))
        )
        return Decision(option.id, rationale, usage)

    def attend(self, facts: tuple[Fact, ...]) -> tuple[Fact, ...]:
        if len(facts) <= self.context_window:
            return facts
        tail = self.context_window - self.pinned
        return facts[: self.pinned] + facts[-tail:]

    @staticmethod
    def score(option: Option, facts: tuple[Fact, ...]) -> float:
        """Facts support an option only if they mention its subject; shared refs amplify and
        exonerating facts count against it."""
        total = 0.0
        for f in facts:
            if option.subject not in f.refs:
                continue
            sign = -1.0 if option.subject in f.exonerates else 1.0
            total += sign * f.salience * len(f.refs & option.refs)
        return total

    def _may_choose(self, option: Option, score: float, request: DecisionRequest) -> bool:
        if not option.commits:
            return True
        only_commits = all(o.commits for o in request.options)
        return only_commits or score >= self.commit_threshold
