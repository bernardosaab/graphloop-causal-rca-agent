# Domain glossary

| Term | Meaning |
|---|---|
| **Causal Context Graph (CCG)** | One typed graph holding the context (services, dependencies, changes, incidents, evidence) and the causal layer (fault modes that caused symptoms). |
| **Fault mode** | A (service, fault type) pair that can be a root cause, e.g. `bad_deploy@payments`. Also called a hypothesis or a `Diagnosis`. |
| **Symptom service** | The service that raised the alert. Usually a victim, not the cause. |
| **Blast radius** | The symptom service and everything it transitively depends on: the only places a root cause can be. |
| **Victim** | A service that shows errors only because a dependency is failing. Its logs point downstream (`upstream_timeout`). |
| **Signature** | Evidence only the faulty component emits, e.g. a stack trace in a new build. |
| **Red herring** | A recent change or anomaly unrelated to the root cause. |
| **Probe** | One budgeted tool call (`get_logs`, `get_metrics`, `get_dependencies`) on one service. |
| **Verify loop** | The bounded sub-graph that selects and runs probes and updates the belief. |
| **Sensor** | A deterministic check after a step, e.g. the critique's evidence rule or the counterfactual check. |
| **Recipe** | The compiled, model-free path for a corroborated causal pattern. |
| **Escalation** | The agent declines to commit and hands the incident to a person with its suggestion. |
| **Proposal** | The exact remediation payload derived from a committed diagnosis. |
| **Risk tier** | soft, hard or critical, set by blast radius; it decides who must sign. |
| **Fingerprint** | SHA-256 of the proposal payload; the signature is bound to it. |
| **Read-back** | Checking the system of record after a write; the source of truth for "resolved". |
| **Causal assertion** | One bi-temporal record that a fault mode caused a symptom, with valid and transaction time. |
| **Corroborated** | A causal link supported by at least 3 verified incidents. |
| **Degraded mode** | A rule-based fallback taken, and traced, when a cap or outage stops the model. |
