"""The Causal Context Graph (CCG).

A single typed multigraph that unifies two views usually kept apart:

* the **context graph**: services, their dependencies, change events, incidents and the
  evidence observed while investigating them;
* the **causal graph**: ``CAUSES`` edges from a *fault mode* (e.g. ``bad_deploy@payments``)
  to the symptoms it produced.

Causal knowledge is **bi-temporal and append-only**. Every confirmed root cause is stored as a
:class:`CausalAssertion` with *valid time* (when the cause was true in the world) and
*transaction time* (when we recorded it, and when, if ever, we stopped believing it).
Nothing is deleted: a wrong assertion is *invalidated*, so the graph can always answer
"what did we believe at decision time?" (``as_of``) as well as "what do we believe now?".
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

import networkx as nx

from graphloop.domain.topology import Topology
from graphloop.domain.vocabulary import APPLICABLE_FAULTS, Diagnosis, Finding


class NodeKind(StrEnum):
    SERVICE = "service"
    FAULT_MODE = "fault_mode"
    CHANGE = "change"
    INCIDENT = "incident"
    EVIDENCE = "evidence"


class EdgeKind(StrEnum):
    # context edges
    DEPENDS_ON = "DEPENDS_ON"
    HAS_FAULT_MODE = "HAS_FAULT_MODE"
    CHANGED = "CHANGED"
    AFFECTED = "AFFECTED"
    OBSERVED_ON = "OBSERVED_ON"
    PART_OF = "PART_OF"
    # causal edges
    CAUSES = "CAUSES"
    ROOT_CAUSE_OF = "ROOT_CAUSE_OF"


CAUSAL_EDGES = frozenset({EdgeKind.CAUSES, EdgeKind.ROOT_CAUSE_OF})


class Provenance(StrEnum):
    LEARNED = "learned"  # supported by at least one verified incident
    CORROBORATED = "corroborated"  # supported by CORROBORATION_THRESHOLD or more


CORROBORATION_THRESHOLD = 3


@dataclass(frozen=True, slots=True)
class CausalAssertion:
    """One immutable, bi-temporal record that ``fault`` caused a symptom on ``symptom``."""

    assertion_id: int
    incident_id: str
    fault: Diagnosis
    symptom: str
    confidence: float
    evidence: tuple[str, ...]
    source: str  # which component wrote it (single-writer audit)
    valid_from: int  # valid time: when the cause was observed in the world
    recorded_at: int  # transaction time: when it entered the graph
    invalidated_at: int | None = None  # transaction time: when it stopped being believed
    invalidation_reason: str | None = None

    def believed_at(self, t: int) -> bool:
        return self.recorded_at <= t and (self.invalidated_at is None or self.invalidated_at > t)


@dataclass(frozen=True, slots=True)
class CausalLink:
    """Aggregate view over the assertions believed at some point in time."""

    fault: Diagnosis
    symptom: str
    support: int
    confidence: float
    provenance: Provenance
    last_seen: int


class CausalContextStore(Protocol):
    """Port used by agents. Swap the networkx adapter for Neo4j/PuppyGraph behind this."""

    @property
    def topology(self) -> Topology: ...

    def blast_radius(self, symptom: str) -> frozenset[str]: ...

    def explain_path(self, symptom: str, root: str) -> tuple[str, ...] | None: ...

    def add_change(
        self, change_id: str, service: str, kind: str, minutes_ago: int, description: str
    ) -> None: ...

    def open_incident(self, incident_id: str, symptom: str, at: int) -> None: ...

    def add_evidence(self, incident_id: str, finding: Finding) -> str: ...

    def fault_prior(
        self, hypothesis: Diagnosis, as_of: int | None = None, window: int | None = None
    ) -> float: ...

    def causal_link(
        self, fault: Diagnosis, symptom: str, as_of: int | None = None
    ) -> CausalLink | None: ...

    def record_cause(
        self,
        incident_id: str,
        diagnosis: Diagnosis,
        symptom: str,
        confidence: float,
        evidence_ids: tuple[str, ...],
        at: int,
        source: str,
    ) -> CausalAssertion: ...

    def invalidate(self, incident_id: str, at: int, reason: str) -> list[CausalAssertion]: ...


def service_id(name: str) -> str:
    return f"svc:{name}"


def fault_id(d: Diagnosis) -> str:
    return f"fault:{d.fault.value}@{d.service}"


class NetworkXCausalContextGraph:
    """In-memory CCG backed by a ``networkx.MultiDiGraph``."""

    def __init__(self, topology: Topology) -> None:
        self._topology = topology
        self._g: nx.MultiDiGraph[str] = nx.MultiDiGraph()
        self._assertions: list[CausalAssertion] = []
        self._evidence_seq = 0
        self._clock = 0
        for svc in topology.services:
            self._g.add_node(
                service_id(svc.name),
                kind=NodeKind.SERVICE,
                label=svc.name,
                service_kind=svc.kind.value,
            )
            for fault in APPLICABLE_FAULTS[svc.kind]:
                d = Diagnosis(svc.name, fault)
                self._g.add_node(fault_id(d), kind=NodeKind.FAULT_MODE, label=str(d))
                self._g.add_edge(
                    service_id(svc.name),
                    fault_id(d),
                    key=EdgeKind.HAS_FAULT_MODE,
                    kind=EdgeKind.HAS_FAULT_MODE,
                )
        for caller, callee in topology.dependencies:
            self._g.add_edge(
                service_id(caller),
                service_id(callee),
                key=EdgeKind.DEPENDS_ON,
                kind=EdgeKind.DEPENDS_ON,
            )

    # ---- read side: context --------------------------------------------------------------
    @property
    def topology(self) -> Topology:
        return self._topology

    @property
    def graph(self) -> nx.MultiDiGraph[str]:
        """Read-only view for visualisation and analytics."""
        return self._g.copy(as_view=True)

    @property
    def clock(self) -> int:
        return self._clock

    def blast_radius(self, symptom: str) -> frozenset[str]:
        """Every component whose failure could explain a symptom on ``symptom``."""
        return frozenset({symptom}) | self._topology.downstream(symptom)

    def explain_path(self, symptom: str, root: str) -> tuple[str, ...] | None:
        return self._topology.path(symptom, root)

    def incident_subgraph(self, incident_id: str) -> nx.MultiDiGraph[str]:
        """The incident, its evidence and changes, and the blast radius they sit in."""
        inc = f"inc:{incident_id}"
        if inc not in self._g:
            raise KeyError(f"unknown incident {incident_id!r}")
        nodes: set[str] = {inc}
        for pred in self._g.predecessors(inc):
            nodes.add(pred)
            # A fault mode may be the root cause of other incidents too: keep only this one.
            nodes.update(
                s
                for s in self._g.successors(pred)
                if self._g.nodes[s]["kind"] is not NodeKind.INCIDENT
            )
        nodes.update(self._g.successors(inc))
        services = {n for n in nodes if self._g.nodes[n]["kind"] is NodeKind.SERVICE}
        for svc in list(services):
            nodes.update(service_id(s) for s in self.blast_radius(self._g.nodes[svc]["label"]))
        for node in list(nodes):
            if self._g.nodes[node]["kind"] is NodeKind.SERVICE:
                nodes.update(
                    p
                    for p in self._g.predecessors(node)
                    if self._g.nodes[p]["kind"] is NodeKind.CHANGE
                    and p.startswith(f"chg:{incident_id}:")
                )
        return self._g.subgraph(nodes).copy()

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, data in self._g.nodes(data=True):
            counts[str(data["kind"])] = counts.get(str(data["kind"]), 0) + 1
        counts["causal_assertions"] = len(self._assertions)
        counts["believed_assertions"] = len(self.assertions())
        return counts

    # ---- read side: bi-temporal causal memory ----------------------------------------------
    def assertions(self, as_of: int | None = None) -> list[CausalAssertion]:
        t = self._clock if as_of is None else as_of
        return [a for a in self._assertions if a.believed_at(t)]

    def history(self) -> list[CausalAssertion]:
        """Every assertion ever recorded, including invalidated ones."""
        return list(self._assertions)

    def causal_links(self, as_of: int | None = None, window: int | None = None) -> list[CausalLink]:
        grouped: dict[tuple[Diagnosis, str], list[CausalAssertion]] = {}
        for a in self._believed(as_of, window):
            grouped.setdefault((a.fault, a.symptom), []).append(a)
        links = [_aggregate(fault, symptom, group) for (fault, symptom), group in grouped.items()]
        return sorted(links, key=lambda link: (-link.support, str(link.fault), link.symptom))

    def causal_link(
        self, fault: Diagnosis, symptom: str, as_of: int | None = None
    ) -> CausalLink | None:
        group = [
            a for a in self._believed(as_of, None) if a.fault == fault and a.symptom == symptom
        ]
        return _aggregate(fault, symptom, group) if group else None

    def fault_prior(
        self, hypothesis: Diagnosis, as_of: int | None = None, window: int | None = None
    ) -> float:
        """Prior weight ``1 + support``; ``window`` keeps only recent valid-time evidence."""
        if fault_id(hypothesis) not in self._g:
            return 0.0
        support = sum(1 for a in self._believed(as_of, window) if a.fault == hypothesis)
        return 1.0 + support

    # ---- write side ----------------------------------------------------------------------
    def add_change(
        self, change_id: str, service: str, kind: str, minutes_ago: int, description: str
    ) -> None:
        node = f"chg:{change_id}"
        self._g.add_node(
            node,
            kind=NodeKind.CHANGE,
            label=f"{kind}: {description}",
            change_kind=kind,
            minutes_ago=minutes_ago,
        )
        self._g.add_edge(node, service_id(service), key=EdgeKind.CHANGED, kind=EdgeKind.CHANGED)

    def open_incident(self, incident_id: str, symptom: str, at: int) -> None:
        self._clock = max(self._clock, at)
        node = f"inc:{incident_id}"
        self._g.add_node(node, kind=NodeKind.INCIDENT, label=f"incident {incident_id}", at=at)
        self._g.add_edge(node, service_id(symptom), key=EdgeKind.AFFECTED, kind=EdgeKind.AFFECTED)

    def add_evidence(self, incident_id: str, finding: Finding) -> str:
        self._evidence_seq += 1
        node = f"ev:{incident_id}:{self._evidence_seq}"
        self._g.add_node(
            node,
            kind=NodeKind.EVIDENCE,
            label=finding.signal.value,
            text=finding.text,
            signal=finding.signal.value,
        )
        self._g.add_edge(
            node, service_id(finding.service), key=EdgeKind.OBSERVED_ON, kind=EdgeKind.OBSERVED_ON
        )
        self._g.add_edge(node, f"inc:{incident_id}", key=EdgeKind.PART_OF, kind=EdgeKind.PART_OF)
        return node

    def record_cause(
        self,
        incident_id: str,
        diagnosis: Diagnosis,
        symptom: str,
        confidence: float,
        evidence_ids: tuple[str, ...],
        at: int,
        source: str,
    ) -> CausalAssertion:
        if fault_id(diagnosis) not in self._g:
            raise KeyError(f"{diagnosis} is not a valid fault mode")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        self._clock = max(self._clock, at)
        assertion = CausalAssertion(
            assertion_id=len(self._assertions),
            incident_id=incident_id,
            fault=diagnosis,
            symptom=symptom,
            confidence=confidence,
            evidence=evidence_ids,
            source=source,
            valid_from=at,
            recorded_at=at,
        )
        self._assertions.append(assertion)
        self._g.add_edge(
            fault_id(diagnosis),
            f"inc:{incident_id}",
            key=EdgeKind.ROOT_CAUSE_OF,
            kind=EdgeKind.ROOT_CAUSE_OF,
            confidence=confidence,
        )
        self._sync_causes_edge(diagnosis, symptom)
        return assertion

    def invalidate(self, incident_id: str, at: int, reason: str) -> list[CausalAssertion]:
        """Stop believing every assertion written for ``incident_id``. History is kept."""
        self._clock = max(self._clock, at)
        changed: list[CausalAssertion] = []
        for i, a in enumerate(self._assertions):
            if a.incident_id == incident_id and a.invalidated_at is None:
                self._assertions[i] = replace(a, invalidated_at=at, invalidation_reason=reason)
                changed.append(self._assertions[i])
        for a in changed:
            self._sync_causes_edge(a.fault, a.symptom)
        return changed

    # ---- internals -----------------------------------------------------------------------
    def _believed(self, as_of: int | None, window: int | None) -> list[CausalAssertion]:
        t = self._clock if as_of is None else as_of
        return [
            a
            for a in self._assertions
            if a.believed_at(t) and (window is None or a.valid_from > t - window)
        ]

    def _sync_causes_edge(self, fault: Diagnosis, symptom: str) -> None:
        """Keep the materialised ``CAUSES`` edge equal to the currently believed aggregate."""
        src, dst = fault_id(fault), service_id(symptom)
        link = self.causal_link(fault, symptom)
        if self._g.has_edge(src, dst, key=EdgeKind.CAUSES):
            self._g.remove_edge(src, dst, key=EdgeKind.CAUSES)
        if link is not None:
            self._g.add_edge(
                src,
                dst,
                key=EdgeKind.CAUSES,
                kind=EdgeKind.CAUSES,
                support=link.support,
                confidence=link.confidence,
                provenance=link.provenance.value,
            )


def _aggregate(fault: Diagnosis, symptom: str, group: list[CausalAssertion]) -> CausalLink:
    support = len(group)
    return CausalLink(
        fault=fault,
        symptom=symptom,
        support=support,
        confidence=sum(a.confidence for a in group) / support,
        provenance=(
            Provenance.CORROBORATED if support >= CORROBORATION_THRESHOLD else Provenance.LEARNED
        ),
        last_seen=max(a.valid_from for a in group),
    )
