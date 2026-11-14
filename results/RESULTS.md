# Benchmark results

_Generated 2026-11-14 02:13 UTC by `graphloop bench` · 100 incidents x 1 repeat(s) · model `simulated` · seed 7 · budget 12 probes._

Every agent receives the same incident stream, alert, service catalog, tools, probe budget and model seed (a paired design). A run counts as **resolved** only when its remediation passed the fail-closed gate, executed, and the read-back confirmed the incident was fixed.

| Agent | Resolved (95% CI) | Precision | Coverage | Wrong writes | Escalated | Tool calls | LLM calls | Tokens | Cost / resolved | Compiled path | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `loop` | 56% [46%, 65%] | 56% | 100% | 44 | 0 | 2.7 | 3.7 | 7,220 | $0.00201 | 0% | 17 ms |
| `graph` | 77% [68%, 84%] | 100% | 77% | 0 | 23 | 3.1 | 3.6 | 1,112 | $0.00027 | 0% | 30 ms |
| `graph+memory` | 80% [71%, 87%] | 100% | 80% | 0 | 20 | 3.0 | 2.9 | 900 | $0.00021 | 30% | 32 ms |

![resolved](img/resolved.svg) ![wrong writes](img/wrong_writes.svg)

![tokens](img/tokens.svg) ![learning](img/learning_llm.svg)

## Paired significance

Exact McNemar test, `loop` vs `graph+memory` on identical incidents: 6 incidents only the loop resolved, 30 only the graph agent resolved, **p = 7e-05**.

## Error analysis

| Agent | Failure clusters |
|---|---|
| `loop` | `confounded_by_change` x 41, `wrong_fault_type` x 3 |
| `graph` | `escalated` x 23 |
| `graph+memory` | `escalated` x 20 |

**Retrieval vs generation.** Root cause in the graph agent's top-3 hypotheses (recall@3): `graph` 100%, `graph+memory` 100%. Misses after a retrieval hit are therefore decision or evidence problems, not retrieval problems.

## One incident, two architectures (`inc-70003`)

Ground truth: `resource_exhaustion@catalog-db`. The loop agent concluded `bad_deploy@api-gateway`; the graph agent concluded `resource_exhaustion@catalog-db`.

**loop**

```mermaid
flowchart TB
  %% loop
  s0["reason<br/><small>LLM: get_logs(web-frontend)</small>"]
  s1["act<br/><small>get_logs(web-frontend)</small>"]
  s0 --> s1
  s2["reason<br/><small>LLM: get_dependencies(api-gateway)</small>"]
  s1 --> s2
  s3["act<br/><small>get_dependencies(api-gateway)</small>"]
  s2 --> s3
  s4["reason<br/><small>LLM: get_logs(api-gateway)</small>"]
  s3 --> s4
  s5["act<br/><small>get_logs(api-gateway)</small>"]
  s4 --> s5
  s6["reason<br/><small>LLM: get_metrics(api-gateway)</small>"]
  s5 --> s6
  s7["act<br/><small>get_metrics(api-gateway)</small>"]
  s6 --> s7
  s8["reason<br/><small>=> bad_deploy@api-gateway</small>"]
  s7 --> s8
```

**graph+memory**

```mermaid
flowchart TB
  %% graph+memory
  s0["triage"]
  s1["hypothesize"]
  s0 --> s1
  s2["recipe"]
  s1 --> s2
  s3["select_probe<br/><small>LLM: get_logs(api-gateway) - tests bad_deploy@api-gateway (p</small>"]
  s2 --> s3
  s4["run_probe<br/><small>get_logs(api-gateway)</small>"]
  s3 --> s4
  s5["select_probe"]
  s4 --> s5
  s6["run_probe<br/><small>get_logs(catalog)</small>"]
  s5 --> s6
  s7["select_probe<br/><small>LLM: get_logs(inventory) - tests bad_deploy@inventory (p=0.0</small>"]
  s6 --> s7
  s8["run_probe<br/><small>get_logs(inventory)</small>"]
  s7 --> s8
  s9["select_probe<br/><small>LLM: get_logs(catalog-db) - tests dependency_outage@catalog-</small>"]
  s8 --> s9
  s10["run_probe<br/><small>get_logs(catalog-db)</small>"]
  s9 --> s10
  s11["select_probe<br/><small>LLM: get_metrics(catalog-db) - tests dependency_outage@catal</small>"]
  s10 --> s11
  s12["run_probe<br/><small>get_metrics(catalog-db)</small>"]
  s11 --> s12
  s13["counterfactual"]
  s12 --> s13
  s14["diagnose<br/><small>LLM: resource_exhaustion on catalog-db (p=0.86)</small>"]
  s13 --> s14
  s15["critique<br/><small>accept on causal chain</small>"]
  s14 --> s15
  s16["finalize<br/><small>=> resource_exhaustion@catalog-db</small>"]
  s15 --> s16
```


## How to read these numbers

* Results with `simulated` are produced by a deterministic, agent-agnostic stand-in for a model (see `docs/adr/0005-simulated-llm-evaluation.md`). They measure the *architecture* under controlled conditions; they are not claims about any specific model. Re-run with `--llm ollama:<model>` for a real open-weight model.
* Cost uses a reference price of $0.15 / $0.60 per million input / output tokens so agents are comparable; local models cost nothing at the margin.
* Escalations are not failures of safety: the agent declined to act without direct evidence and handed the incident to a person with its best suggestion.
