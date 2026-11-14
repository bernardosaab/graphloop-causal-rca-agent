# 0004. Fail-closed writes; score the write, not the claim

* Status: accepted
* Date: 2026-11-13

## Context

A diagnosis is a claim; a remediation is a write to production. An agent should never decide on
its own what gets written, and success should be judged against reality, not against the
agent's confidence.

## Decision

Every agent uses one shared remediation workflow (`governance/workflow.py`):

1. **Propose.** A `ChangeProposal` is derived only from the typed diagnosis, through the `REMEDY`
   table. Free text from logs or alerts cannot widen it, which also defeats instructions planted
   in telemetry (tested with `inject_instruction=True`).
2. **Assess.** The risk tier follows blast radius: soft; hard (3 or more dependents); critical
   (data tier or failover).
3. **Sign.** A versioned policy signs soft changes. Hard and critical changes pause on a
   LangGraph `interrupt` until a named person signs the exact fingerprint they were shown.
4. **Execute.** The gate refuses unless the change is approved *and* the SHA-256 fingerprint of
   the payload matches what was signed. An idempotency key prevents double writes.
5. **Read back.** The system of record says whether the incident is actually resolved.

Every step is appended to a hash-chained decision log.

## Consequences

* The benchmark's headline metric is *resolved by read-back*, and wrong writes are reported
  explicitly.
* Escalated investigations produce no write at all.
