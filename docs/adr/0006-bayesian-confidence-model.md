# 0006. Confidence is a small, inspectable Bayesian model

* Status: accepted
* Date: 2026-11-13

## Context

A model's own reported confidence is not calibrated. We want a confidence that moves for stated
reasons and can be audited.

## Decision

`domain/belief.py` keeps an immutable probability distribution over (service, fault)
hypotheses. Each finding multiplies every hypothesis by a likelihood ratio from one table, and
the distribution is renormalised:

| Evidence | Effect |
|---|---|
| fault signature on the service (e.g. a stack trace after a deploy) | x12 for the matching fault, x0.3 for other faults on that service |
| healthy telemetry on the service | x0.05 |
| the service times out calling a callee | x0.15 for that service; x3 inside the callee's subtree, x0.2 elsewhere |
| a supporting change (deploy, config) | x3 for the matching fault |
| CPU and memory saturation | x6 for resource exhaustion |
| a generic error or latency spike | x1.5 |

Priors come from the CCG: `1 + verified support`, optionally limited to a valid-time window.

## Consequences

* Every posterior can be explained. The update rule has unit tests, and property tests check that
  the result is always a probability distribution.
* The table is a modelling choice. Changing it is a reviewed change, gated by the benchmark.
