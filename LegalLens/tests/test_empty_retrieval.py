from rag_pipeline import LegalLensPipeline
from utils.models import RAGAnswer


def test_answer_handles_empty_retrieval_without_llm(settings_override) -> None:
    pipeline = LegalLensPipeline(settings_override)
    answer = pipeline.answer("What is the notice period?", top_k=3, retrieved=[])

    assert isinstance(answer, RAGAnswer)
    assert "could not find" in answer.answer.lower()
    assert answer.citations == []

