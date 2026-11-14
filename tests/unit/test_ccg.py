from __future__ import annotations

import pytest

from graphloop.domain.ccg import (
    EdgeKind,
    NetworkXCausalContextGraph,
    Provenance,
    fault_id,
    service_id,
)
from graphloop.domain.topology import Topology
from graphloop.domain.vocabulary import Diagnosis, FaultType, Finding, Signal

PAY_DEPLOY = Diagnosis("pay", FaultType.BAD_DEPLOY)


def _record(ccg: NetworkXCausalContextGraph, inc: str, at: int, d: Diagnosis = PAY_DEPLOY) -> None:
    ccg.open_incident(inc, "web", at=at)
    ccg.record_cause(inc, d, "web", 0.9, (), at=at, source="test")


def test_seeds_context_from_catalog(chain: Topology) -> None:
    ccg = NetworkXCausalContextGraph(chain)
    g = ccg.graph
    assert g.has_edge(service_id("api"), service_id("pay"), key=EdgeKind.DEPENDS_ON)
    assert fault_id(Diagnosis("db", FaultType.DEPENDENCY_OUTAGE)) in g
    # Fault modes are typed by component kind: a frontend cannot have a datastore outage.
    assert fault_id(Diagnosis("web", FaultType.DEPENDENCY_OUTAGE)) not in g


def test_blast_radius_and_explanations(chain: Topology) -> None:
    ccg = NetworkXCausalContextGraph(chain)
    assert ccg.blast_radius("web") == {"web", "gw", "api", "pay", "inv", "db"}
    assert "lonely" not in ccg.blast_radius("web")
    assert ccg.explain_path("web", "lonely") is None


def test_support_accumulates_into_corroborated_links(chain: Topology) -> None:
    ccg = NetworkXCausalContextGraph(chain)
    assert ccg.fault_prior(PAY_DEPLOY) == 1.0
    for t in range(3):
        _record(ccg, f"i{t}", t)
    link = ccg.causal_link(PAY_DEPLOY, "web")
    assert link is not None
    assert link.support == 3
    assert link.provenance is Provenance.CORROBORATED
    assert ccg.fault_prior(PAY_DEPLOY) == 4.0
    assert (
        ccg.graph.get_edge_data(fault_id(PAY_DEPLOY), service_id("web"), key=EdgeKind.CAUSES)[
            "support"
        ]
        == 3
    )


def test_bitemporal_as_of_and_invalidation_keep_history(chain: Topology) -> None:
    ccg = NetworkXCausalContextGraph(chain)
    _record(ccg, "i1", at=1)
    _record(ccg, "i2", at=2)
    invalidated = ccg.invalidate("i2", at=5, reason="read-back regressed")
    assert [a.incident_id for a in invalidated] == ["i2"]
    # Now we believe one assertion; at t=3 we believed two. Nothing was deleted.
    assert len(ccg.assertions()) == 1
    assert len(ccg.assertions(as_of=3)) == 2
    assert len(ccg.history()) == 2
    assert ccg.history()[1].invalidation_reason == "read-back regressed"
    assert ccg.fault_prior(PAY_DEPLOY, as_of=3) == 3.0
    assert ccg.fault_prior(PAY_DEPLOY) == 2.0


def test_valid_time_window_forgets_old_evidence(chain: Topology) -> None:
    ccg = NetworkXCausalContextGraph(chain)
    _record(ccg, "old", at=1)
    _record(ccg, "new", at=50)
    assert ccg.fault_prior(PAY_DEPLOY) == 3.0
    assert ccg.fault_prior(PAY_DEPLOY, window=10) == 2.0


def test_invalidating_every_assertion_removes_the_materialised_edge(chain: Topology) -> None:
    ccg = NetworkXCausalContextGraph(chain)
    _record(ccg, "i1", at=1)
    ccg.invalidate("i1", at=2, reason="wrong")
    assert ccg.causal_link(PAY_DEPLOY, "web") is None
    assert not ccg.graph.has_edge(fault_id(PAY_DEPLOY), service_id("web"), key=EdgeKind.CAUSES)


def test_rejects_invalid_writes(chain: Topology) -> None:
    ccg = NetworkXCausalContextGraph(chain)
    ccg.open_incident("i1", "web", at=0)
    with pytest.raises(KeyError, match="not a valid fault mode"):
        ccg.record_cause(
            "i1", Diagnosis("web", FaultType.DEPENDENCY_OUTAGE), "web", 0.5, (), at=0, source="t"
        )
    with pytest.raises(ValueError, match="confidence"):
        ccg.record_cause("i1", PAY_DEPLOY, "web", 1.5, (), at=0, source="t")
    assert ccg.fault_prior(Diagnosis("web", FaultType.DEPENDENCY_OUTAGE)) == 0.0


def test_incident_subgraph_contains_evidence_and_changes(chain: Topology) -> None:
    ccg = NetworkXCausalContextGraph(chain)
    ccg.open_incident("i1", "web", at=0)
    ccg.add_change("i1:c0", "pay", "deploy", 5, "deployed pay v2")
    ev = ccg.add_evidence("i1", Finding("pay", Signal.CODE_EXCEPTION, "NPE", 0.75))
    sub = ccg.incident_subgraph("i1")
    assert {ev, "chg:i1:c0", service_id("pay"), service_id("db")} <= set(sub.nodes)
    assert ccg.stats()["evidence"] == 1
    with pytest.raises(KeyError):
        ccg.incident_subgraph("missing")
