from retrieval.retriever import SemanticRetriever
from utils.models import Chunk, RetrievedChunk


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2]


class FakeStore:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        self.calls += 1
        return [
            RetrievedChunk(
                chunk=Chunk("id", "contract.pdf", 2, "Notice", "Thirty days notice.", 0, 4),
                score=0.9,
            )
        ]


def test_retriever_returns_scores_and_calls_store_once() -> None:
    store = FakeStore()
    retriever = SemanticRetriever(FakeEmbedder(), store)  # type: ignore[arg-type]

    results = retriever.retrieve("notice period", top_k=1)

    assert store.calls == 1
    assert results[0].score == 0.9
    assert results[0].chunk.page_number == 2

