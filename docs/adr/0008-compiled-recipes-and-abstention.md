# 0008. Repeated work compiles to a model-free recipe; unknown cases escalate

* Status: accepted
* Date: 2026-11-13

## Context

Most incidents repeat. Paying for model judgement on a known pattern is waste, and guessing on an
unknown one is risk.

## Decision

* **Recipe path.** A recipe fires when the change feed shows a deploy or config change on a
  service, and that fault mode has a *corroborated* causal link to this symptom (3 or more
  verified incidents). The agent then runs one deterministic probe. If the probe finds the fault
  signature, the agent diagnoses without calling the model. If not, the miss is recorded and the
  run falls through to the agentic loop.
* **Abstention.** The critique sensor accepts a diagnosis in two cases. One is when the fault
  signature was observed. The other is a complete causal chain: the timeout trail ends at the
  service, the service shows its own errors, and the posterior is at least 0.85. Otherwise it
  re-probes within its bounds and then escalates to a person with its best suggestion.
  Escalations are reported as a separate outcome, not hidden.

## Consequences

* In the reference benchmark about 30% of incidents take the compiled path, and model calls per
  incident fall over the incident stream.
* About 20% of incidents escalate. Almost all are cases where the evidence cannot separate two
  fault types on the same service; guessing there would execute the wrong remediation.
* Widening exploration to competing services was tried. It cost 40% more tokens with no gain in
  accuracy, so it was reverted.
