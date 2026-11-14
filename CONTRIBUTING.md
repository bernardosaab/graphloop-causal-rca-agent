# Contributing

```bash
uv sync
uv run pre-commit install
make check        # ruff, ruff format --check, mypy --strict, pytest with an 85% coverage gate
```

* Write the test first. Tests for agent behaviour assert on trajectories and stop paths, not
  only on the final answer.
* The benchmark is a release gate (`tests/integration/test_benchmark.py`). A change to the agent
  must not introduce wrong writes or lose the significant win over the baseline. If it changes
  the headline numbers, regenerate `results/` with `make bench` and explain why in the PR.
* Record significant design decisions as an ADR in `docs/adr/`.
* Live-model tests are opt-in: `GRAPHLOOP_OLLAMA=1 uv run pytest -m ollama`.
