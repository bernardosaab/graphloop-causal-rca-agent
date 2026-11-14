from __future__ import annotations

from dataclasses import replace

from graphloop.agents.common import Investigation, Path, Status
from graphloop.domain.vocabulary import Diagnosis, FaultType
from graphloop.eval.taxonomy import FailureMode, classify_failure
from graphloop.simulator.generator import Scenario


def _inv(
    d: Diagnosis | None, status: Status = Status.COMMITTED, exhausted: bool = False
) -> Investigation:
    return Investigation(
        "x", "i", "web", status, d, (), None, Path.LOOP, budget_exhausted=exhausted
    )


def test_failure_clusters(pay_deploy: Scenario) -> None:
    def cls(inv: Investigation) -> str:
        return classify_failure(pay_deploy, inv)

    assert cls(_inv(None, Status.ESCALATED)) == FailureMode.ESCALATED
    assert cls(_inv(Diagnosis("pay", FaultType.CONFIG_DRIFT))) == FailureMode.WRONG_FAULT_TYPE
    assert cls(_inv(Diagnosis("api", FaultType.BAD_DEPLOY))) == FailureMode.CONFOUNDED_BY_CHANGE
    assert cls(_inv(Diagnosis("gw", FaultType.BAD_DEPLOY))) == FailureMode.BLAMED_VICTIM
    assert cls(_inv(Diagnosis("inv", FaultType.BAD_DEPLOY), exhausted=True)) == (
        FailureMode.BUDGET_EXHAUSTED
    )
    assert cls(_inv(Diagnosis("inv", FaultType.BAD_DEPLOY))) == FailureMode.OTHER


def test_noisy_signal(pay_deploy: Scenario) -> None:
    sc = replace(pay_deploy, changes=())
    assert classify_failure(sc, _inv(Diagnosis("lonely", FaultType.RESOURCE_EXHAUSTION))) == (
        FailureMode.NOISY_SIGNAL
    )
