"""Node implementations for the graph agent.

Design rule: anything that can be computed deterministically from the Causal Context Graph
(blast radius, priors, Bayesian updates, counterfactual checks, evidence critique, known
recipes) *is* computed. The model is consulted only for genuine judgement calls between a
few graph-ranked options with a compact, focused context, and the run never depends on it:
if a cap or outage stops the model, a rule-based choice is taken and flagged as degraded.
"""

from __future__ import annotations

from typing import Any

from graphloop.agents.common import (
    call_tool,
    decide_or_fallback,
    finding_to_fact,
    parse_probe,
    probe_key,
)
from graphloop.agents.graph_agent.state import GraphAgentConfig, GraphState
from graphloop.domain.belief import BeliefState
from graphloop.domain.ccg import CausalContextStore, Provenance
from graphloop.domain.vocabulary import (
    APPLICABLE_FAULTS,
    SIGNATURE_OF,
    Diagnosis,
    FaultType,
    Finding,
    Signal,
    ToolName,
)
from graphloop.llm.base import DecisionRequest, Fact, LLMClient, Option
from graphloop.simulator.environment import IncidentEnvironment
from graphloop.tracing.tracer import EventKind, Tracer

_PROBE_ORDER = (ToolName.GET_LOGS, ToolName.GET_METRICS)
_CHANGE_FAULT = {
    Signal.RECENT_DEPLOY: FaultType.BAD_DEPLOY,
    Signal.RECENT_CONFIG_CHANGE: FaultType.CONFIG_DRIFT,
}

SELECT_INSTRUCTIONS = (
    "Pick the diagnostic probe that best tests the leading root-cause hypotheses. "
    "Hypothesis probabilities come from a causal model over the service graph."
)
DIAGNOSE_INSTRUCTIONS = (
    "Pick the root cause best supported by the evidence and the hypothesis probabilities."
)


class GraphAgentNodes:
    """Each public method is a LangGraph node or router: ``(state) -> partial update``."""

    def __init__(
        self,
        ccg: CausalContextStore,
        env: IncidentEnvironment,
        llm: LLMClient,
        tracer: Tracer,
        config: GraphAgentConfig,
    ) -> None:
        self._ccg = ccg
        self._env = env
        self._llm = llm
        self._tracer = tracer
        self._cfg = config

    # ---- outer workflow ----------------------------------------------------------------
    def triage(self, state: GraphState) -> dict[str, Any]:  # noqa: ARG002
        self._enter("triage")
        alert = self._env.alert()
        incident_id = self._env.scenario.scenario_id
        self._ccg.open_incident(incident_id, alert.service, at=self._env.at)
        for change in alert.changes:
            self._ccg.add_change(
                f"{incident_id}:{change.change_id}",
                change.service,
                change.kind,
                change.minutes_ago,
                change.description,
            )
        evidence = [self._ccg.add_evidence(incident_id, f) for f in alert.findings]
        return {
            "incident_id": incident_id,
            "symptom": alert.service,
            "findings": list(alert.findings),
            "evidence_ids": evidence,
            "probed": [],
            "iterations": 0,
            "reflections": 0,
            "rejections": 0,
            "forced_probe": None,
            "next_probe": None,
            "exhausted": False,
            "recipe": None,
            "hallucinated": 0,
        }

    def hypothesize(self, state: GraphState) -> dict[str, Any]:
        """Graph-RAG retrieval: bound the search to the blast radius, seed with learned priors."""
        self._enter("hypothesize")
        topo = self._ccg.topology
        radius = self._ccg.blast_radius(state["symptom"])
        weights = {
            d: self._ccg.fault_prior(d, window=self._cfg.memory_window)
            for svc in sorted(radius)
            for d in (Diagnosis(svc, f) for f in APPLICABLE_FAULTS[topo.service(svc).kind])
        }
        relevant = [f for f in state["findings"] if f.service in radius]
        pruned = sorted({f.service for f in state["findings"] if f.service not in radius})
        if pruned:
            self._tracer.emit(
                EventKind.NOTE,
                "counterfactual_prune",
                services=pruned,
                reason="no dependency path from the symptom",
            )
        belief = BeliefState.from_weights(weights).update(relevant, topo)
        self._emit_belief("hypothesize", belief)
        return {"belief": belief}

    def recipe(self, state: GraphState) -> dict[str, Any]:
        """Compiled path: a corroborated causal pattern plus a matching change => one probe.

        Repeated work should not need a model. If the recipe's check fails, that is recorded
        (a miss is data) and the run falls through to the agentic verification loop.
        """
        self._enter("recipe")
        match = self._matching_recipe(state) if self._cfg.enable_recipes else None
        if match is None:
            return {"recipe": None}
        hypothesis, support = match
        name = f"{hypothesis}->{state['symptom']}"
        call = call_tool(self._env, self._tracer, "recipe", ToolName.GET_LOGS, hypothesis.service)
        update: dict[str, Any] = {
            "probed": [*state["probed"], probe_key(ToolName.GET_LOGS, hypothesis.service)],
        }
        if call is None:
            return update | {"recipe": None}
        update |= self._absorb(state, call.findings)
        if self._has_signature(hypothesis, list(call.findings)):
            self._tracer.emit(EventKind.NOTE, "recipe_hit", recipe=name, support=support)
            belief: BeliefState = update["belief"]
            return update | {
                "recipe": name,
                "diagnosis": hypothesis,
                "confidence": belief.posteriors[hypothesis],
            }
        self._tracer.emit(EventKind.NOTE, "recipe_miss", recipe=name, support=support)
        return update | {"recipe": None}

    def counterfactual(self, state: GraphState) -> dict[str, Any]:
        """Reject the leader if the observed graph state contradicts it being the cause."""
        self._enter("counterfactual")
        best, _ = state["belief"].best
        reason = self._contradiction(best, state)
        if reason is None:
            return {"verdict": "consistent"}
        self._tracer.emit(
            EventKind.NOTE, "counterfactual_reject", hypothesis=str(best), reason=reason
        )
        belief = state["belief"].eliminate(best)
        self._emit_belief("counterfactual", belief)
        return {"belief": belief, "rejections": state["rejections"] + 1, "verdict": "rejected"}

    def diagnose(self, state: GraphState) -> dict[str, Any]:
        self._enter("diagnose")
        top = state["belief"].top(self._cfg.top_k)
        options = tuple(
            Option(
                str(d),
                f"{d.fault.value} on {d.service} (p={p:.2f})",
                d.service,
                frozenset({d.service, d.fault.value}),
                commits=True,
            )
            for d, p in top
        )
        choice = options[0].id
        if len(options) > 1:
            request = DecisionRequest(
                "diagnose", DIAGNOSE_INSTRUCTIONS, self._focused_facts(state), options
            )
            choice = decide_or_fallback(self._llm, request, options[0].id, self._tracer)
        diagnosis = next(d for d, _ in top if str(d) == choice)
        return {"diagnosis": diagnosis, "confidence": state["belief"].posteriors[diagnosis]}

    def critique(self, state: GraphState) -> dict[str, Any]:
        """Deterministic sensor: a diagnosis must be backed by its own failure signature."""
        self._enter("critique")
        diagnosis = state["diagnosis"]
        assert diagnosis is not None
        if self._has_signature(diagnosis, state["findings"]):
            return {"verdict": "accept"}
        if self._chain_complete(diagnosis, state):
            self._tracer.emit(
                EventKind.NOTE,
                "accept_on_causal_chain",
                diagnosis=str(diagnosis),
                confidence=round(state["confidence"], 3),
            )
            return {"verdict": "accept"}
        probe = self._next_unprobed(diagnosis.service, state)
        can_retry = (
            state["reflections"] < self._cfg.max_reflections
            and probe is not None
            and self._env.remaining > 0
        )
        if not can_retry:
            self._tracer.emit(
                EventKind.NOTE,
                "escalate",
                diagnosis=str(diagnosis),
                reason="no direct evidence for the leading hypothesis",
            )
            return {"verdict": "unverified"}
        self._tracer.emit(EventKind.NOTE, "critique_revise", diagnosis=str(diagnosis), probe=probe)
        return {
            "verdict": "revise",
            "forced_probe": probe,
            "exhausted": False,
            "reflections": state["reflections"] + 1,
        }

    def finalize(self, state: GraphState) -> dict[str, Any]:
        self._enter("finalize")
        self._tracer.emit(
            EventKind.DIAGNOSIS,
            "diagnosis",
            diagnosis=str(state["diagnosis"]),
            confidence=round(state["confidence"], 3),
            verdict=state["verdict"],
            path="compiled" if state.get("recipe") else "agentic",
        )
        return {}

    # ---- inner verification loop ---------------------------------------------------------
    def select_probe(self, state: GraphState) -> dict[str, Any]:
        self._enter("select_probe")
        if (forced := state.get("forced_probe")) is not None:
            return {"next_probe": forced, "forced_probe": None}
        candidates = self._candidate_probes(state)
        if not candidates:
            return {"next_probe": None, "exhausted": True}
        if len(candidates) == 1:
            return {"next_probe": candidates[0][0]}
        options = tuple(
            Option(key, f"{key.replace(':', '(')}) - {why}", parse_probe(key)[1], refs)
            for key, why, refs in candidates
        )
        request = DecisionRequest(
            "select_probe", SELECT_INSTRUCTIONS, self._focused_facts(state), options
        )
        return {"next_probe": decide_or_fallback(self._llm, request, options[0].id, self._tracer)}

    def run_probe(self, state: GraphState) -> dict[str, Any]:
        self._enter("run_probe")
        key = state.get("next_probe")
        if key is None:
            return {"iterations": state["iterations"] + 1}
        tool, svc = parse_probe(key)
        call = call_tool(self._env, self._tracer, "run_probe", tool, svc)
        update: dict[str, Any] = {
            "probed": [*state["probed"], key],
            "iterations": state["iterations"] + 1,
            "next_probe": None,
        }
        if call is None:
            return update | {"hallucinated": state["hallucinated"] + 1}
        return update | self._absorb(state, call.findings)

    # ---- routers -------------------------------------------------------------------------
    def should_verify(self, state: GraphState) -> bool:
        if state.get("forced_probe") is not None:
            return self._env.remaining > 0
        if self._env.remaining <= 0 or state["iterations"] >= self._cfg.max_verify_iterations:
            return False
        if state.get("exhausted", False):  # selector found nothing left to test
            return False
        return state["belief"].best[1] < self._cfg.confidence_threshold

    @staticmethod
    def after_recipe(state: GraphState) -> str:
        return "hit" if state.get("recipe") else "miss"

    def after_counterfactual(self, state: GraphState) -> str:
        if state["verdict"] == "rejected" and state["rejections"] <= self._cfg.max_rejections:
            return "verify"
        return "diagnose"

    @staticmethod
    def after_critique(state: GraphState) -> str:
        return "verify" if state["verdict"] == "revise" else "finalize"

    # ---- helpers -------------------------------------------------------------------------
    def _absorb(self, state: GraphState, findings: tuple[Finding, ...]) -> dict[str, Any]:
        new_ids = [self._ccg.add_evidence(state["incident_id"], f) for f in findings]
        belief = state["belief"].update(findings, self._ccg.topology)
        self._emit_belief("evidence", belief)
        return {
            "belief": belief,
            "findings": [*state["findings"], *findings],
            "evidence_ids": [*state["evidence_ids"], *new_ids],
        }

    def _matching_recipe(self, state: GraphState) -> tuple[Diagnosis, int] | None:
        radius = self._ccg.blast_radius(state["symptom"])
        best: tuple[Diagnosis, int] | None = None
        for f in state["findings"]:
            fault = _CHANGE_FAULT.get(f.signal)
            if fault is None or f.service not in radius:
                continue
            hypothesis = Diagnosis(f.service, fault)
            link = self._ccg.causal_link(hypothesis, state["symptom"])
            if link is None or link.provenance is not Provenance.CORROBORATED:
                continue
            if best is None or link.support > best[1]:
                best = (hypothesis, link.support)
        return best

    def _candidate_probes(self, state: GraphState) -> list[tuple[str, str, frozenset[str]]]:
        probed = set(state["probed"])
        out: list[tuple[str, str, frozenset[str]]] = []

        def add(key: str, why: str, refs: frozenset[str]) -> None:
            if key not in probed and all(key != k for k, _, _ in out):
                out.append((key, why, refs))

        for d, p in state["belief"].top(self._cfg.top_k):
            nxt = self._next_unprobed(d.service, state)
            if nxt is not None:
                add(
                    nxt,
                    f"tests {d} (p={p:.2f})",
                    frozenset({d.service, d.fault.value, parse_probe(nxt)[0].value}),
                )
        # Follow the evidence trail: the newest victim->callee pointer not yet inspected.
        logs = ToolName.GET_LOGS.value
        for f in reversed(state["findings"]):
            if f.signal is Signal.UPSTREAM_TIMEOUT and f.target:
                add(
                    probe_key(ToolName.GET_LOGS, f.target),
                    f"follows timeout trail from {f.service}",
                    frozenset({f.target, logs}),
                )
                break
        else:
            add(
                probe_key(ToolName.GET_LOGS, state["symptom"]),
                "inspects the alerting service",
                frozenset({state["symptom"], logs}),
            )
        return out

    def _next_unprobed(self, service: str, state: GraphState) -> str | None:
        probed = set(state["probed"])
        return next(
            (probe_key(t, service) for t in _PROBE_ORDER if probe_key(t, service) not in probed),
            None,
        )

    def _focused_facts(self, state: GraphState) -> tuple[Fact, ...]:
        hyps = [
            Fact(
                f"hypothesis {d}: probability {p:.2f}",
                frozenset({d.service, d.fault.value}),
                2.0 * p,
            )
            for d, p in state["belief"].top(5)
        ]
        recent = [finding_to_fact(f) for f in state["findings"][-4:]]
        return tuple(hyps + recent)

    def _contradiction(self, h: Diagnosis, state: GraphState) -> str | None:
        on_h = [f for f in state["findings"] if f.service == h.service]
        if any(f.signal is Signal.UPSTREAM_TIMEOUT for f in on_h):
            return f"{h.service} is itself waiting on a downstream dependency"
        if any(f.signal is Signal.HEALTHY for f in on_h):
            return f"{h.service} reports healthy telemetry"
        path = self._ccg.explain_path(state["symptom"], h.service)
        if path is None:
            return f"no dependency path from {state['symptom']} to {h.service}"
        healthy = {f.service for f in state["findings"] if f.signal is Signal.HEALTHY}
        if broken := [s for s in path[1:-1] if s in healthy]:
            return f"path {' -> '.join(path)} passes through healthy {broken[0]}"
        return None

    def _chain_complete(self, d: Diagnosis, state: GraphState) -> bool:
        """The trail of timeouts ends at ``d`` and ``d`` itself shows its own errors."""
        threshold = self._cfg.chain_acceptance
        if threshold is None or state["confidence"] < threshold:
            return False
        on_d = [f for f in state["findings"] if f.service == d.service]
        pointed_at = any(
            f.signal is Signal.UPSTREAM_TIMEOUT and f.target == d.service for f in state["findings"]
        )
        own_errors = any(f.signal in (Signal.ERROR_SPIKE, Signal.CPU_SATURATION) for f in on_d)
        victim = any(f.signal is Signal.UPSTREAM_TIMEOUT for f in on_d)
        return pointed_at and own_errors and not victim

    @staticmethod
    def _has_signature(d: Diagnosis, findings: list[Finding]) -> bool:
        return any(
            f.service == d.service and SIGNATURE_OF.get(f.signal) is d.fault for f in findings
        )

    def _enter(self, node: str) -> None:
        self._tracer.emit(EventKind.NODE, node)

    def _emit_belief(self, node: str, belief: BeliefState) -> None:
        self._tracer.emit(
            EventKind.BELIEF,
            node,
            top=[(str(d), round(p, 3)) for d, p in belief.top(3)],
            entropy=round(belief.entropy(), 3),
        )
