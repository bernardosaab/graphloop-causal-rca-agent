"""Immutable service topology: who calls whom."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from graphloop.domain.vocabulary import ServiceKind


@dataclass(frozen=True, slots=True)
class Service:
    name: str
    kind: ServiceKind


@dataclass(frozen=True)
class Topology:
    """A directed acyclic call graph; an edge ``(caller, callee)`` means caller needs callee."""

    services: tuple[Service, ...]
    dependencies: tuple[tuple[str, str], ...]
    _by_name: dict[str, Service] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_name = {s.name: s for s in self.services}
        if len(by_name) != len(self.services):
            raise ValueError("service names must be unique")
        for caller, callee in self.dependencies:
            if caller not in by_name or callee not in by_name:
                raise ValueError(f"unknown service in dependency {caller}->{callee}")
            if caller == callee:
                raise ValueError(f"self dependency on {caller}")
        object.__setattr__(self, "_by_name", by_name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.services)

    def service(self, name: str) -> Service:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown service {name!r}") from exc

    def callees(self, name: str) -> tuple[str, ...]:
        return tuple(sorted(c for a, c in self.dependencies if a == name))

    def callers(self, name: str) -> tuple[str, ...]:
        return tuple(sorted(a for a, c in self.dependencies if c == name))

    def downstream(self, name: str) -> frozenset[str]:
        """All transitive callees of ``name`` (excluding itself)."""
        return self._closure(name, self.callees)

    def upstream(self, name: str) -> frozenset[str]:
        """All transitive callers of ``name`` (excluding itself)."""
        return self._closure(name, self.callers)

    def path(self, source: str, target: str) -> tuple[str, ...] | None:
        """Shortest call path ``source -> ... -> target`` or ``None``."""
        if source == target:
            return (source,)
        parents: dict[str, str] = {}
        queue: deque[str] = deque([source])
        seen = {source}
        while queue:
            node = queue.popleft()
            for nxt in self.callees(node):
                if nxt in seen:
                    continue
                parents[nxt] = node
                if nxt == target:
                    hops = [target]
                    while hops[-1] != source:
                        hops.append(parents[hops[-1]])
                    return tuple(reversed(hops))
                seen.add(nxt)
                queue.append(nxt)
        return None

    @staticmethod
    def _closure(start: str, step: Callable[[str], tuple[str, ...]]) -> frozenset[str]:
        seen: set[str] = set()
        stack = list(step(start))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(step(node))
        return frozenset(seen)
