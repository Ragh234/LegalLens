from dataclasses import dataclass
from time import perf_counter

from prompts.templates import build_prompt_parts
from retrieval.retriever import SemanticRetriever
from utils.models import RetrievedChunk


@dataclass(frozen=True)
class EvaluationTrace:
    question: str
    retrieved_chunks: list[RetrievedChunk]
    final_prompt: str
    latency_ms: float


class RAGEvaluator:
    """Expose retrieval internals for learning, debugging, and interview demos."""

    def __init__(self, retriever: SemanticRetriever) -> None:
        self.retriever = retriever

    def inspect(self, question: str, top_k: int, retrieved: list[RetrievedChunk] | None = None) -> EvaluationTrace:
        start = perf_counter()
        retrieved = retrieved if retrieved is not None else self.retriever.retrieve(question, top_k)
        prompt = build_prompt_parts(question, retrieved).final_prompt
        return EvaluationTrace(
            question=question,
            retrieved_chunks=retrieved,
            final_prompt=prompt,
            latency_ms=(perf_counter() - start) * 1000,
        )
