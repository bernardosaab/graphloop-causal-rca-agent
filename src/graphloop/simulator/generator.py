"""Seeded generators for service topologies and incident scenarios.

Determinism is a feature: the same seed always produces the same topology, fault,
red herrings and telemetry, which makes benchmark runs reproducible and diffable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from graphloop.domain.topology import Service, Topology
from graphloop.domain.vocabulary import APPLICABLE_FAULTS, Diagnosis, FaultType, ServiceKind

_FRONTENDS = ("web-frontend",)
_GATEWAYS = ("api-gateway",)
_TIER1 = ("checkout", "catalog", "search", "user-profile", "orders")
_TIER2 = ("payments", "inventory", "pricing", "auth", "shipping", "recommendations")
_DATASTORES = ("orders-db", "users-db", "catalog-db", "redis-cache")


def generate_topology(seed: int) -> Topology:
    """A layered microservice topology: frontend -> gateway -> tier1 -> tier2 -> datastores."""
    rng = random.Random(seed)
    tier1 = sorted(rng.sample(_TIER1, 4))
    tier2 = sorted(rng.sample(_TIER2, 4))
    stores = sorted(rng.sample(_DATASTORES, 3))

    services = (
        [Service(n, ServiceKind.FRONTEND) for n in _FRONTENDS]
        + [Service(n, ServiceKind.GATEWAY) for n in _GATEWAYS]
        + [Service(n, ServiceKind.SERVICE) for n in (*tier1, *tier2)]
        + [Service(n, ServiceKind.DATASTORE) for n in stores]
    )
    deps: set[tuple[str, str]] = {(_FRONTENDS[0], _GATEWAYS[0])}
    deps.update((_GATEWAYS[0], t) for t in tier1)
    for t in tier1:
        deps.update((t, c) for c in rng.sample(tier2, rng.randint(1, 2)))
        if rng.random() < 0.5:
            deps.add((t, rng.choice(stores)))
    for t in tier2:
        deps.add((t, rng.choice(stores)))
    # Guarantee every tier2 service and datastore has at least one caller.
    for t in tier2:
        if not any(c == t for _, c in deps):
            deps.add((rng.choice(tier1), t))
    for s in stores:
        if not any(c == s for _, c in deps):
            deps.add((rng.choice(tier2), s))
    return Topology(tuple(services), tuple(sorted(deps)))


@dataclass(frozen=True, slots=True)
class ChangeEvent:
    change_id: str
    service: str
    kind: str  # "deploy" | "config"
    minutes_ago: int
    description: str


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    topology: Topology
    root_cause: Diagnosis
    alert_service: str
    changes: tuple[ChangeEvent, ...]
    noisy_services: tuple[str, ...]
    affected: frozenset[str] = frozenset()

    @property
    def depth(self) -> int:
        path = self.topology.path(self.alert_service, self.root_cause.service)
        return len(path) - 1 if path else 0


def fault_modes(topology: Topology) -> list[Diagnosis]:
    """Every (service, fault) pair that can be a root cause (frontend and gateway excluded)."""
    return [
        Diagnosis(s.name, f)
        for s in topology.services
        if s.kind in (ServiceKind.SERVICE, ServiceKind.DATASTORE)
        for f in APPLICABLE_FAULTS[s.kind]
    ]


def hotspot_propensity(topology: Topology, seed: int, hotspots: int = 3) -> dict[Diagnosis, float]:
    """A skewed fault distribution: a few fault modes account for most incidents."""
    rng = random.Random(seed)
    modes = fault_modes(topology)
    hot = set(rng.sample(modes, hotspots))
    return {m: (8.0 if m in hot else 1.0) for m in modes}


class ScenarioGenerator:
    """Samples incidents on a fixed topology from a fault propensity distribution."""

    def __init__(
        self,
        topology: Topology,
        propensity: dict[Diagnosis, float] | None = None,
        red_herrings: tuple[int, int] = (2, 3),
    ) -> None:
        self._topology = topology
        self._propensity = propensity or dict.fromkeys(fault_modes(topology), 1.0)
        self._red_herrings = red_herrings

    @property
    def topology(self) -> Topology:
        return self._topology

    def generate(self, seed: int) -> Scenario:
        rng = random.Random(seed)
        topo = self._topology
        modes = sorted(self._propensity)
        root = rng.choices(modes, weights=[self._propensity[m] for m in modes], k=1)[0]
        alert = next(
            s.name
            for s in topo.services
            if s.kind is ServiceKind.FRONTEND and root.service in topo.downstream(s.name)
        )
        affected = topo.upstream(root.service) & (topo.downstream(alert) | {alert})

        changes: list[ChangeEvent] = []
        if root.fault is FaultType.BAD_DEPLOY:
            changes.append(_deploy(rng, root.service, "c0", rng.randint(5, 30)))
        elif root.fault is FaultType.CONFIG_DRIFT:
            changes.append(_config(rng, root.service, "c0", rng.randint(5, 30)))

        others = [n for n in topo.names if n not in {root.service, alert}]
        n_herrings = rng.randint(*self._red_herrings)
        # Bias red herrings toward victims on the failure path: the classic confounder.
        victims = sorted(affected - {alert})
        for i in range(n_herrings):
            pool = victims if victims and rng.random() < 0.6 else others
            svc = rng.choice(pool)
            maker = _deploy if rng.random() < 0.6 else _config
            changes.append(maker(rng, svc, f"c{i + 1}", rng.randint(2, 40)))
        rng.shuffle(changes)

        noisy_pool = [n for n in others if n not in affected]
        noisy = tuple(rng.sample(noisy_pool, 1)) if noisy_pool else ()

        return Scenario(
            scenario_id=f"inc-{seed:05d}",
            seed=seed,
            topology=topo,
            root_cause=root,
            alert_service=alert,
            changes=tuple(sorted(changes, key=lambda c: c.minutes_ago)),
            noisy_services=noisy,
            affected=frozenset(affected),
        )


def _deploy(rng: random.Random, service: str, cid: str, minutes: int) -> ChangeEvent:
    version = f"v{rng.randint(1, 4)}.{rng.randint(0, 30)}.{rng.randint(0, 9)}"
    return ChangeEvent(cid, service, "deploy", minutes, f"deployed {service} {version}")


def _config(rng: random.Random, service: str, cid: str, minutes: int) -> ChangeEvent:
    key = rng.choice(("pool.max_connections", "timeout_ms", "feature.flags", "retry.budget"))
    return ChangeEvent(cid, service, "config", minutes, f"changed {key} on {service}")
