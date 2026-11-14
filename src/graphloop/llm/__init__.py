from graphloop.llm.base import Decision, DecisionRequest, Fact, LLMClient, Option, TokenUsage
from graphloop.llm.ollama import LLMUnavailableError, OllamaClient
from graphloop.llm.simulated import SimulatedLLM


def build_llm(spec: str, seed: int = 0) -> LLMClient:
    """Factory: ``simulated`` or ``ollama[:<model>]``."""
    if spec == "simulated":
        return SimulatedLLM(seed=seed)
    if spec == "ollama" or spec.startswith("ollama:"):
        _, _, model = spec.partition(":")
        return OllamaClient(model=model) if model else OllamaClient()
    raise ValueError(f"unknown LLM spec {spec!r} (expected 'simulated' or 'ollama[:model]')")


__all__ = [
    "Decision",
    "DecisionRequest",
    "Fact",
    "LLMClient",
    "LLMUnavailableError",
    "OllamaClient",
    "Option",
    "SimulatedLLM",
    "TokenUsage",
    "build_llm",
]
