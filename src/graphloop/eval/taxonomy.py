"""Error analysis: put every failed run into one named cluster, so the biggest gets fixed first.

The clusters come from reading failed traces (open coding) and grouping them by cause (axial
coding). The rules are checked in order; the first match wins.
"""

from __future__ import annotations

from enum import StrEnum

from graphloop.agents.common import Investigation, Status
from graphloop.simulator.generator import Scenario


class FailureMode(StrEnum):
    ESCALATED = "escalated"  # declined to guess: safe, but not an accepted output
    WRONG_FAULT_TYPE = "wrong_fault_type"  # right component, wrong failure mode
    CONFOUNDED_BY_CHANGE = "confounded_by_change"  # blamed a recently changed bystander
    BLAMED_VICTIM = "blamed_victim"  # blamed a service that was only propagating errors
    NOISY_SIGNAL = "noisy_signal"  # chased an unrelated anomaly
    BUDGET_EXHAUSTED = "budget_exhausted"  # ran out of probes before converging
    OTHER = "other"


def classify_failure(scenario: Scenario, inv: Investigation) -> str:
    if inv.status is Status.ESCALATED:
        return FailureMode.ESCALATED.value
    d = inv.committed
    if d is None:  # pragma: no cover - committed implies a diagnosis
        return FailureMode.OTHER.value
    if d.service == scenario.root_cause.service:
        return FailureMode.WRONG_FAULT_TYPE.value
    if d.service in {c.service for c in scenario.changes}:
        return FailureMode.CONFOUNDED_BY_CHANGE.value
    if d.service in scenario.affected:
        return FailureMode.BLAMED_VICTIM.value
    if d.service in scenario.noisy_services:
        return FailureMode.NOISY_SIGNAL.value
    if inv.budget_exhausted:
        return FailureMode.BUDGET_EXHAUSTED.value
    return FailureMode.OTHER.value
