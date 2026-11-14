# 0005. Evaluate with a simulated, agent-agnostic model, and say what that means

* Status: accepted
* Date: 2026-11-13

## Context

We want reproducible, free comparisons of two *architectures* that can run in CI. Real models
are non-deterministic, cost money or hardware, and change between versions.

## Decision

`llm/simulated.py` is a deterministic, seeded stand-in for a model. It answers through the same
constrained-decision interface as a real model and simulates three documented failure modes:

* **Salience-driven attention.** An option is scored by the visible facts that mention its
  subject, weighted by how alarming each looks. Facts can also exonerate: a caller that times
  out on X argues against the caller.
* **Limited effective context.** Only the first and the most recent facts are attended to.
* **Slips.** With small probability it picks a runner-up option.

It must see a threshold of support before ending an investigation while other actions remain.
It never sees which agent is calling or the ground truth. Both agents turn observations into
facts with the same function (`agents.common.finding_to_fact`), and the runner gives both the
same model seed for each incident (a paired design).

## Calibration, disclosed

* At a commit threshold of 2.5 the baseline scored about 20%, which made it a strawman. At 3.0 it
  follows the timeout trail as a reasonable engineer would, and scores 56-70% depending on the
  incident mix.
* Without log sampling the graph agent scored 98-100%, which was not credible. The simulator now
  drops the root's error line 25% of the time, for every agent.

## Threats to validity

* The reasoner is a model of failure modes, not a language model. Absolute numbers say nothing
  about any specific model; the *gaps* between agents show what the architecture contributes
  under shared conditions.
* The same authors wrote the graph agent's likelihood ratios and the simulator. The ratios are a
  small, documented table (ADR 0006), and release gates catch regressions.
* External validity comes from `--llm ollama:<model>`. Publish those runs next to the simulated
  ones, labelled as measured and with the model named.
