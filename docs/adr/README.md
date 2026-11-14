# Architecture decision records

| # | Decision |
|---|---|
| [0001](0001-langgraph-as-mechanism.md) | LangGraph is the mechanism; the harness policy lives in our code |
| [0002](0002-unified-causal-context-graph.md) | One Causal Context Graph, behind a store port |
| [0003](0003-bounded-loops-inside-the-graph.md) | Loops live inside the graph, each with a hard bound |
| [0004](0004-fail-closed-writes.md) | Fail-closed writes; score the write, not the claim |
| [0005](0005-simulated-llm-evaluation.md) | Evaluate with a simulated, agent-agnostic model, and say what that means |
| [0006](0006-bayesian-confidence-model.md) | Confidence is a small, inspectable Bayesian model |
| [0007](0007-bitemporal-memory-from-verified-outcomes.md) | Causal memory is bi-temporal and written only from verified outcomes |
| [0008](0008-compiled-recipes-and-abstention.md) | Repeated work compiles to a model-free recipe; unknown cases escalate |
| [0009](0009-durable-execution.md) | Durable execution with bounded resume and a deserialisation allow-list |
