from __future__ import annotations

import pytest

from graphloop.eval.stats import mcnemar_exact, pass_hat_k, percentile, wilson


def test_wilson_interval() -> None:
    p = wilson(8, 10)
    assert p.rate == 0.8
    assert p.low == pytest.approx(0.490, abs=1e-3)
    assert p.high == pytest.approx(0.943, abs=1e-3)
    assert wilson(0, 0).high == 1.0
    assert wilson(10, 10).high == 1.0
    assert "80%" in str(p)
    with pytest.raises(ValueError, match="successes"):
        wilson(3, 2)


def test_mcnemar_exact() -> None:
    a = [True] * 10 + [False] * 10
    b = [True] * 10 + [True] * 10
    result = mcnemar_exact(a, b)
    assert (result.b, result.c) == (0, 10)
    assert result.p_value == pytest.approx(2 / 1024)
    assert mcnemar_exact([True], [True]).p_value == 1.0
    with pytest.raises(ValueError, match="equal length"):
        mcnemar_exact([True], [])


def test_pass_hat_k_and_percentiles() -> None:
    assert pass_hat_k([[True, True], [True, False], [True, True]]) == pytest.approx(2 / 3)
    assert pass_hat_k([]) == 0.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5.0], 95) == 5.0
    assert percentile([], 50) == 0.0
