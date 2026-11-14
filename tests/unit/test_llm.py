from __future__ import annotations

import json
import os

import httpx
import pytest

from graphloop.llm import LLMUnavailableError, OllamaClient, SimulatedLLM, build_llm
from graphloop.llm.base import DecisionRequest, Fact, Option
from graphloop.llm.prompt import render_prompt, response_schema


def _fact(text: str, refs: set[str], sal: float, exonerates: set[str] | None = None) -> Fact:
    return Fact(text, frozenset(refs), sal, frozenset(exonerates or set()))


PROBE_A = Option("logs:a", "logs(a)", "a", frozenset({"a", "logs"}))
PROBE_B = Option("logs:b", "logs(b)", "b", frozenset({"b", "logs"}))
SUBMIT_A = Option("submit:a", "submit a", "a", frozenset({"a", "bad_deploy"}), commits=True)


def _request(facts: list[Fact], options: list[Option]) -> DecisionRequest:
    return DecisionRequest("test", "pick", tuple(facts), tuple(options))


def test_request_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _request([], [])
    with pytest.raises(ValueError, match="unique"):
        _request([], [PROBE_A, PROBE_A])
    with pytest.raises(KeyError):
        _request([], [PROBE_A]).option("zzz")


def test_scores_only_facts_about_the_subject_and_exoneration() -> None:
    llm = SimulatedLLM(slip_rate=0.0)
    facts = (_fact("a errors", {"a"}, 1.0), _fact("b deployed", {"b", "bad_deploy"}, 0.8))
    assert llm.score(PROBE_A, facts) == 1.0
    assert llm.score(SUBMIT_A, facts) == 1.0  # b's deploy fact does not support a
    victim = (_fact("a timed out calling b", {"a", "b"}, 0.85, {"a"}),)
    assert llm.score(PROBE_A, victim) < 0 < llm.score(PROBE_B, victim)


def test_commit_threshold_prefers_investigating() -> None:
    llm = SimulatedLLM(slip_rate=0.0, commit_threshold=3.0)
    weak = [_fact("a deployed", {"a", "bad_deploy"}, 0.8)]
    assert llm.decide(_request(weak, [PROBE_A, SUBMIT_A])).option_id == "logs:a"
    strong = [*weak, _fact("a NPE", {"a", "bad_deploy"}, 0.9)]
    assert llm.decide(_request(strong, [PROBE_A, SUBMIT_A])).option_id == "submit:a"
    # With only commit options left, it must still answer.
    assert llm.decide(_request(weak, [SUBMIT_A])).option_id == "submit:a"


def test_limited_context_drops_the_middle() -> None:
    llm = SimulatedLLM(context_window=4, pinned=1)
    facts = tuple(_fact(str(i), {str(i)}, 1.0) for i in range(10))
    assert [f.text for f in llm.attend(facts)] == ["0", "7", "8", "9"]
    with pytest.raises(ValueError, match="context_window"):
        SimulatedLLM(context_window=2, pinned=2)
    with pytest.raises(ValueError, match="slip_rate"):
        SimulatedLLM(slip_rate=2.0)


def test_slips_are_seeded_and_reproducible() -> None:
    facts = [_fact("a", {"a"}, 1.0)]
    req = _request(facts, [PROBE_A, PROBE_B])
    runs = [
        [SimulatedLLM(seed=3, slip_rate=0.5).decide(req).option_id for _ in range(1)]
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    picks = {SimulatedLLM(seed=s, slip_rate=1.0).decide(req).option_id for s in range(5)}
    assert picks == {"logs:b"}  # always slips to the runner-up


def test_prompt_and_schema_constrain_answers() -> None:
    req = _request([_fact("a errors", {"a"}, 1.0)], [PROBE_A, PROBE_B])
    text = render_prompt(req)
    assert "[logs:a]" in text
    assert "a errors" in text
    assert response_schema(req)["properties"]["option_id"]["enum"] == ["logs:a", "logs:b"]  # type: ignore[index]


def _ollama(responses: list[dict[str, object]] | Exception) -> OllamaClient:
    queue = list(responses) if isinstance(responses, list) else []

    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(responses, Exception):
            raise responses
        return httpx.Response(200, json=queue.pop(0))

    return OllamaClient(transport=httpx.MockTransport(handler))


def _chat(option_id: str) -> dict[str, object]:
    return {
        "message": {"content": json.dumps({"option_id": option_id, "rationale": "r"})},
        "prompt_eval_count": 100,
        "eval_count": 10,
    }


def test_ollama_parses_valid_answers() -> None:
    d = _ollama([_chat("logs:b")]).decide(_request([], [PROBE_A, PROBE_B]))
    assert (d.option_id, d.valid, d.usage.prompt_tokens) == ("logs:b", True, 100)


def test_ollama_retries_then_flags_invalid_output() -> None:
    client = _ollama([_chat("not-an-option"), {"message": {"content": "garbage"}}])
    d = client.decide(_request([], [PROBE_A, PROBE_B]))
    assert d.valid is False
    assert d.option_id == "logs:a"


def test_ollama_outage_is_a_typed_error() -> None:
    client = _ollama(httpx.ConnectError("refused"))
    with pytest.raises(LLMUnavailableError):
        client.decide(_request([], [PROBE_A]))


def test_factory() -> None:
    assert isinstance(build_llm("simulated"), SimulatedLLM)
    assert build_llm("ollama:llama3.2").name == "ollama:llama3.2"
    with pytest.raises(ValueError, match="unknown LLM"):
        build_llm("gpt")


@pytest.mark.ollama
@pytest.mark.skipif(not os.environ.get("GRAPHLOOP_OLLAMA"), reason="set GRAPHLOOP_OLLAMA=1")
def test_live_ollama_contract() -> None:  # pragma: no cover - needs a local server
    d = OllamaClient().decide(_request([_fact("a errors", {"a"}, 1.0)], [PROBE_A, PROBE_B]))
    assert d.option_id in {"logs:a", "logs:b"}
