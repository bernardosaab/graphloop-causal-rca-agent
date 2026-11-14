# 0003. Loops live inside the graph, each with a hard bound

* Status: accepted
* Date: 2026-11-13

## Context

A pure tool loop lets the model decide when it is done. That is cheap to build and expensive to
run: it over-probes, it commits early on salient but wrong evidence, and its cost grows with the
transcript.

## Decision

The outer workflow is a fixed graph. Iteration happens in only three places. Each is bounded in
code and ends on a typed condition:

| Loop | Continues while | Hard bound |
|---|---|---|
| verify (sub-graph) | best posterior < `confidence_threshold`, and budget and probes remain | `max_verify_iterations`, probe budget |
| counterfactual -> verify | the leading hypothesis contradicts the observed graph state | `max_rejections` |
| critique -> verify | the diagnosis has no direct evidence | `max_reflections` |

The model is asked only two kinds of question: "which of these few graph-ranked probes?" and
"which of the top-3 hypotheses?". Its context is a compact list of hypotheses and the latest
evidence.

## Consequences

* Context stays small and roughly constant: about 1k tokens per incident, against about 7k for
  the loop baseline.
* Every stop path is testable (`tests/integration/test_agents.py`).
* The loops can still be wrong. That is what the critique sensor and the write gate are for.
