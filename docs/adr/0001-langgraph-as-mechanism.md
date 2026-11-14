# 0001. LangGraph is the mechanism; the harness policy lives in our code

* Status: accepted
* Date: 2026-11-13

## Context

We need typed state, conditional edges, sub-graphs, checkpointing and human-in-the-loop pauses.
We could build a small runtime ourselves or use LangGraph.

## Decision

Use LangGraph for the *mechanism*: StateGraph, conditional edges, compiled sub-graphs, the
checkpointer and `interrupt`. Keep every *policy* in plain, tested Python that LangGraph only
calls:

* stop conditions: `GraphAgentNodes.should_verify`;
* caps: `harness.gateway.ModelGateway`;
* resume limits: `harness.durable.invoke_durably`;
* approval tiers: `governance.policy`;
* the write gate: `governance.gate.WriteGate`.

Nodes are code. The model is called from inside nodes and is never asked to route the graph.

## Consequences

* A framework inside a harness, not instead of one: replacing LangGraph touches the wiring
  (`agents/*/agent.py`, `governance/workflow.py`), not the policies.
* State must be checkpoint-serialisable, which shaped the domain types (ADR 0009).
* We inherit LangGraph's typing quirks; two call sites need an explicit `RunnableConfig`.

## Alternatives considered

* **Own runtime.** Full control, but it re-implements checkpointing and interrupts, which are
  not the interesting part of this project.
* **A framework where the model routes.** Rejected: routing by prompt makes control flow
  advice rather than a gate.
