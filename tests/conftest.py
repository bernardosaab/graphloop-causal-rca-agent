from __future__ import annotations

import pytest

from graphloop.domain.topology import Service, Topology
from graphloop.domain.vocabulary import Diagnosis, FaultType, ServiceKind
from graphloop.simulator.generator import ChangeEvent, Scenario, generate_topology


@pytest.fixture
def chain() -> Topology:
    """web -> gw -> api -> {pay, inv}; pay -> db; inv -> db."""
    return Topology(
        services=(
            Service("web", ServiceKind.FRONTEND),
            Service("gw", ServiceKind.GATEWAY),
            Service("api", ServiceKind.SERVICE),
            Service("pay", ServiceKind.SERVICE),
            Service("inv", ServiceKind.SERVICE),
            Service("db", ServiceKind.DATASTORE),
            Service("lonely", ServiceKind.SERVICE),
        ),
        dependencies=(
            ("web", "gw"),
            ("gw", "api"),
            ("api", "pay"),
            ("api", "inv"),
            ("pay", "db"),
            ("inv", "db"),
        ),
    )


def make_scenario(
    topology: Topology,
    root: Diagnosis,
    changes: tuple[ChangeEvent, ...] = (),
    noisy: tuple[str, ...] = (),
    seed: int = 1,
) -> Scenario:
    alert = "web"
    affected = topology.upstream(root.service) & (topology.downstream(alert) | {alert})
    return Scenario(
        scenario_id=f"inc-test-{seed}",
        seed=seed,
        topology=topology,
        root_cause=root,
        alert_service=alert,
        changes=changes,
        noisy_services=noisy,
        affected=frozenset(affected),
    )


@pytest.fixture
def pay_deploy(chain: Topology) -> Scenario:
    """bad_deploy on pay, with a red-herring deploy on the victim ``api``."""
    return make_scenario(
        chain,
        Diagnosis("pay", FaultType.BAD_DEPLOY),
        changes=(
            ChangeEvent("c0", "pay", "deploy", 12, "deployed pay v2"),
            ChangeEvent("c1", "api", "deploy", 4, "deployed api v9"),
            ChangeEvent("c2", "lonely", "config", 3, "changed timeout_ms on lonely"),
        ),
        noisy=("lonely",),
    )


@pytest.fixture
def generated_topology() -> Topology:
    return generate_topology(7)
