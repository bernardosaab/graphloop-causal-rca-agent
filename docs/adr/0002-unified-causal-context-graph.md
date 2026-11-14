# 0002. One Causal Context Graph, behind a store port

* Status: accepted
* Date: 2026-11-13

## Context

Incident reasoning needs two kinds of knowledge:

* **Context**: what exists and how it connects (services, dependencies, changes, evidence).
* **Causation**: which fault modes produced which symptoms before.

Kept apart, every query joins them in application code, and the causal layer loses the
provenance the context layer could give it.

## Decision

Model both in one typed multigraph (`domain/ccg.py`):

* node kinds: `service`, `fault_mode`, `change`, `incident`, `evidence`;
* context edges: `DEPENDS_ON`, `CHANGED`, `AFFECTED`, `OBSERVED_ON`, `PART_OF`;
* causal edges: `CAUSES` and `ROOT_CAUSE_OF`.

Agents depend on the `CausalContextStore` protocol. `NetworkXCausalContextGraph` is the
in-memory adapter.

## Consequences

* Blast radius, counterfactual path checks, priors and recipes are graph queries, not prompts.
* networkx keeps the project dependency-light and fast at thousands of nodes. When the graph
  outgrows memory or must be shared across processes, a graph database (Neo4j) or a graph query
  layer over existing tables (PuppyGraph) can implement the same port.
* The protocol is deliberately small: it is the contract a second adapter must honour.
