# 0007. Causal memory is bi-temporal and written only from verified outcomes

* Status: accepted
* Date: 2026-11-13

## Context

Memory that agents write from their own claims amplifies their mistakes. Memory that cannot say
*when* something was believed cannot support audit or replay.

## Decision

* Each confirmed root cause is stored as a `CausalAssertion` with a valid time (`valid_from`)
  and a transaction time (`recorded_at`, `invalidated_at`). Assertions are never deleted: a wrong
  one is invalidated, with a reason. Queries accept `as_of` and an optional valid-time `window`.
* There is a single writer: `GraphAgent.learn`. The harness calls it *after* the remediation
  read-back, and only when the read-back verified the fix. Agents never write memory during a
  run.
* The materialised `CAUSES` edge is a view over the currently believed assertions.

## Consequences

* "What did we believe when we made this decision?" is a query.
* Failed or escalated incidents leave the priors unchanged.
