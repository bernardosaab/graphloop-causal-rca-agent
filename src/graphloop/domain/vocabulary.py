"""Shared, framework-free vocabulary used by every layer.

Everything here is an immutable value object. The simulator *produces* findings,
agents *consume* them, and the Causal Context Graph *stores* them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ServiceKind(StrEnum):
    FRONTEND = "frontend"
    GATEWAY = "gateway"
    SERVICE = "service"
    DATASTORE = "datastore"


class FaultType(StrEnum):
    BAD_DEPLOY = "bad_deploy"
    CONFIG_DRIFT = "config_drift"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_OUTAGE = "dependency_outage"


#: Which faults can originate on which kind of component.
APPLICABLE_FAULTS: dict[ServiceKind, tuple[FaultType, ...]] = {
    ServiceKind.FRONTEND: (FaultType.BAD_DEPLOY, FaultType.CONFIG_DRIFT),
    ServiceKind.GATEWAY: (
        FaultType.BAD_DEPLOY,
        FaultType.CONFIG_DRIFT,
        FaultType.RESOURCE_EXHAUSTION,
    ),
    ServiceKind.SERVICE: (
        FaultType.BAD_DEPLOY,
        FaultType.CONFIG_DRIFT,
        FaultType.RESOURCE_EXHAUSTION,
    ),
    ServiceKind.DATASTORE: (FaultType.RESOURCE_EXHAUSTION, FaultType.DEPENDENCY_OUTAGE),
}


class RemediationAction(StrEnum):
    ROLLBACK_DEPLOY = "rollback_deploy"
    REVERT_CONFIG = "revert_config"
    SCALE_UP = "scale_up"
    FAILOVER = "failover"


#: The one remediation that resolves each fault type.
REMEDY: dict[FaultType, RemediationAction] = {
    FaultType.BAD_DEPLOY: RemediationAction.ROLLBACK_DEPLOY,
    FaultType.CONFIG_DRIFT: RemediationAction.REVERT_CONFIG,
    FaultType.RESOURCE_EXHAUSTION: RemediationAction.SCALE_UP,
    FaultType.DEPENDENCY_OUTAGE: RemediationAction.FAILOVER,
}


class ToolName(StrEnum):
    """Observability tools exposed by the incident environment."""

    GET_RECENT_CHANGES = "get_recent_changes"
    GET_DEPENDENCIES = "get_dependencies"
    GET_METRICS = "get_metrics"
    GET_LOGS = "get_logs"


#: Tools that target a single service (the change log is global).
SERVICE_PROBES: tuple[ToolName, ...] = (
    ToolName.GET_LOGS,
    ToolName.GET_METRICS,
    ToolName.GET_DEPENDENCIES,
)


class Signal(StrEnum):
    """Typed observations a tool can return about a service."""

    HEALTHY = "healthy"
    ERROR_SPIKE = "error_spike"
    LATENCY_SPIKE = "latency_spike"
    CPU_SATURATION = "cpu_saturation"
    UPSTREAM_TIMEOUT = "upstream_timeout"  # victim pointing at a downstream callee
    CODE_EXCEPTION = "code_exception"
    CONFIG_ERROR = "config_error"
    OOM_KILLED = "oom_killed"
    STORE_UNAVAILABLE = "store_unavailable"
    RECENT_DEPLOY = "recent_deploy"
    RECENT_CONFIG_CHANGE = "recent_config_change"
    DEPENDS_ON = "depends_on"


#: Signals that only the faulty component itself emits (its "fingerprint").
SIGNATURE_OF: dict[Signal, FaultType] = {
    Signal.CODE_EXCEPTION: FaultType.BAD_DEPLOY,
    Signal.CONFIG_ERROR: FaultType.CONFIG_DRIFT,
    Signal.OOM_KILLED: FaultType.RESOURCE_EXHAUSTION,
    Signal.STORE_UNAVAILABLE: FaultType.DEPENDENCY_OUTAGE,
}


@dataclass(frozen=True, slots=True)
class Finding:
    """One atomic observation returned by a tool.

    ``target`` is set for relational signals (``UPSTREAM_TIMEOUT``, ``DEPENDS_ON``).
    ``salience`` is how alarming the observation *looks* on the surface: it is
    deliberately independent of whether the observation is actually causal.
    """

    service: str
    signal: Signal
    text: str
    salience: float
    target: str | None = None


@dataclass(frozen=True, slots=True, order=True)
class Diagnosis:
    service: str
    fault: FaultType

    def __str__(self) -> str:
        return f"{self.fault.value}@{self.service}"
