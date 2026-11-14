# 0009. Durable execution with bounded resume and a deserialisation allow-list

* Status: accepted
* Date: 2026-11-13

## Context

Tool calls fail transiently. Restarting a whole investigation wastes budget and can double-write.

## Decision

* Every workflow compiles with a checkpointer (in-memory here; SQLite or Postgres in production)
  and a per-incident `thread_id`.
* `invoke_durably` resumes from the last completed node after a `TransientToolError`, at most
  twice. After that it raises `RunInterruptedError` and records the interruption.
* Graph state holds only serialisable values: strings, numbers, lists and frozen dataclasses.
  `BeliefState` converts lists back to tuples after a checkpoint round trip.
* Checkpoint deserialisation uses an explicit allow-list of our types (`CHECKPOINT_TYPES`) and
  denies everything else.
* Writes carry idempotency keys, so a resumed remediation cannot apply twice.

## Consequences

* Waiting for a person's signature is a checkpointed state, not a blocked thread.
* Nodes must be safe to re-run from the start. Nodes that call tools make their graph writes only
  after the tool returns.
