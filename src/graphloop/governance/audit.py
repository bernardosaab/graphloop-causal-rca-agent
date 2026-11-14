"""Append-only, hash-chained decision log: who proposed, who signed, what ran, what was true."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_GENESIS = "0" * 64


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    seq: int
    incident_id: str
    kind: str  # proposed | approved | declined | executed | duplicate | refused | read_back
    data: dict[str, Any]
    policy_version: str
    prev_hash: str
    hash: str


class DecisionLog:
    def __init__(self) -> None:
        self._records: list[DecisionRecord] = []

    def append(
        self, incident_id: str, kind: str, policy_version: str, **data: Any
    ) -> DecisionRecord:
        prev = self._records[-1].hash if self._records else _GENESIS
        seq = len(self._records)
        body = json.dumps(
            [seq, incident_id, kind, data, policy_version, prev], sort_keys=True, default=str
        )
        record = DecisionRecord(
            seq,
            incident_id,
            kind,
            data,
            policy_version,
            prev,
            hashlib.sha256(body.encode()).hexdigest(),
        )
        self._records.append(record)
        return record

    @property
    def records(self) -> tuple[DecisionRecord, ...]:
        return tuple(self._records)

    def for_incident(self, incident_id: str) -> list[DecisionRecord]:
        return [r for r in self._records if r.incident_id == incident_id]

    def verify_chain(self) -> bool:
        """Detect any edit to a past record."""
        prev = _GENESIS
        for r in self._records:
            body = json.dumps(
                [r.seq, r.incident_id, r.kind, r.data, r.policy_version, prev],
                sort_keys=True,
                default=str,
            )
            if r.prev_hash != prev or hashlib.sha256(body.encode()).hexdigest() != r.hash:
                return False
            prev = r.hash
        return True

    def to_jsonl(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for r in self._records:
                fh.write(json.dumps(asdict(r), default=str) + "\n")
