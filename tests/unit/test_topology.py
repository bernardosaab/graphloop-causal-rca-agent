from __future__ import annotations

import pytest

from graphloop.domain.topology import Service, Topology
from graphloop.domain.vocabulary import ServiceKind


def test_rejects_duplicate_names() -> None:
    svc = Service("a", ServiceKind.SERVICE)
    with pytest.raises(ValueError, match="unique"):
        Topology((svc, svc), ())


def test_rejects_unknown_and_self_dependencies() -> None:
    a = Service("a", ServiceKind.SERVICE)
    with pytest.raises(ValueError, match="unknown"):
        Topology((a,), (("a", "b"),))
    with pytest.raises(ValueError, match="self"):
        Topology((a,), (("a", "a"),))


def test_neighbourhood_queries(chain: Topology) -> None:
    assert chain.callees("api") == ("inv", "pay")
    assert chain.callers("db") == ("inv", "pay")
    assert chain.downstream("gw") == {"api", "pay", "inv", "db"}
    assert chain.upstream("db") == {"web", "gw", "api", "pay", "inv"}
    assert chain.downstream("lonely") == frozenset()


def test_shortest_path(chain: Topology) -> None:
    assert chain.path("web", "db") in {
        ("web", "gw", "api", "pay", "db"),
        ("web", "gw", "api", "inv", "db"),
    }
    assert chain.path("web", "web") == ("web",)
    assert chain.path("db", "web") is None
    assert chain.path("web", "lonely") is None


def test_unknown_service_lookup(chain: Topology) -> None:
    with pytest.raises(KeyError, match="nope"):
        chain.service("nope")
