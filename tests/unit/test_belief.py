from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from graphloop.domain.belief import (
    LR_HEALTHY,
    LR_SIGNATURE_MATCH,
    BeliefState,
    likelihood_ratio,
)
from graphloop.domain.topology import Topology
from graphloop.domain.vocabulary import Diagnosis, FaultType, Finding, Signal

PAY_DEPLOY = Diagnosis("pay", FaultType.BAD_DEPLOY)
PAY_CONFIG = Diagnosis("pay", FaultType.CONFIG_DRIFT)
API_DEPLOY = Diagnosis("api", FaultType.BAD_DEPLOY)
DB_OUTAGE = Diagnosis("db", FaultType.DEPENDENCY_OUTAGE)


def uniform() -> BeliefState:
    return BeliefState.from_weights(
        dict.fromkeys((PAY_DEPLOY, PAY_CONFIG, API_DEPLOY, DB_OUTAGE), 1.0)
    )


def test_from_weights_normalises_and_validates() -> None:
    b = BeliefState.from_weights({PAY_DEPLOY: 3.0, PAY_CONFIG: 1.0})
    assert b.posteriors[PAY_DEPLOY] == pytest.approx(0.75)
    with pytest.raises(ValueError, match="at least one"):
        BeliefState.from_weights({})
    with pytest.raises(ValueError, match="non-negative"):
        BeliefState.from_weights({PAY_DEPLOY: -1.0})


def test_signature_is_strong_evidence_for_its_own_fault(chain: Topology) -> None:
    sig = Finding("pay", Signal.CODE_EXCEPTION, "NPE", 0.75)
    assert likelihood_ratio(PAY_DEPLOY, sig, chain) == LR_SIGNATURE_MATCH
    assert likelihood_ratio(PAY_CONFIG, sig, chain) < 1
    assert likelihood_ratio(API_DEPLOY, sig, chain) < 1
    b = uniform().update([sig], chain)
    assert b.best[0] == PAY_DEPLOY


def test_healthy_telemetry_exonerates(chain: Topology) -> None:
    healthy = Finding("api", Signal.HEALTHY, "fine", -0.5)
    assert likelihood_ratio(API_DEPLOY, healthy, chain) == LR_HEALTHY
    assert uniform().update([healthy], chain).posteriors[API_DEPLOY] < 0.05


def test_timeout_points_down_the_dependency_graph(chain: Topology) -> None:
    pointer = Finding("api", Signal.UPSTREAM_TIMEOUT, "timeout calling pay", 0.85, target="pay")
    b = uniform().update([pointer], chain)
    # api is a victim; pay and its dependency db become more likely.
    assert b.posteriors[API_DEPLOY] < b.posteriors[PAY_DEPLOY]
    assert b.posteriors[DB_OUTAGE] > 0.25


def test_eliminate_and_uniform_fallback() -> None:
    b = BeliefState.from_weights({PAY_DEPLOY: 1.0, PAY_CONFIG: 1.0}).eliminate(PAY_DEPLOY)
    assert b.best == (PAY_CONFIG, 1.0)
    b = b.eliminate(PAY_CONFIG)  # everything ruled out -> maximum ignorance
    assert b.posteriors[PAY_DEPLOY] == pytest.approx(0.5)


def test_ranking_is_deterministic_on_ties() -> None:
    ranked = uniform().ranked()
    assert [d for d, _ in ranked] == sorted(d for d, _ in ranked)
    assert uniform().entropy() == pytest.approx(2.0)


def test_entries_survive_list_round_trip() -> None:
    b = uniform()
    revived = BeliefState([[d, p] for d, p in b.entries])  # type: ignore[arg-type]
    assert revived == b
    assert hash(revived) == hash(b)


_signals = st.sampled_from(list(Signal))
_services = st.sampled_from(["web", "gw", "api", "pay", "inv", "db", "lonely"])


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])  # immutable fixture
@given(st.lists(st.tuples(_services, _signals, st.one_of(st.none(), _services)), max_size=12))
def test_posterior_is_always_a_distribution(
    chain: Topology, raw: list[tuple[str, Signal, str | None]]
) -> None:
    findings = [Finding(s, sig, "x", 0.5, target=t) for s, sig, t in raw]
    b = uniform().update(findings, chain)
    assert math.isclose(sum(b.posteriors.values()), 1.0, rel_tol=1e-9)
    assert all(0.0 <= p <= 1.0 for p in b.posteriors.values())
