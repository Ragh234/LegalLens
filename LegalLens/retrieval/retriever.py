import logging

from embeddings.embedder import BGEEmbedder
from utils.logging import timed_step
from utils.models import RetrievedChunk
from vector_store.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Embeds a question and performs Top-K semantic search over Qdrant."""

    def __init__(self, embedder: BGEEmbedder, vector_store: QdrantVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        with timed_step(logger, f"Semantic retrieval top_k={top_k}"):
            query_vector = self.embedder.embed_query(question)
            chunks = self.vector_store.search(query_vector=query_vector, top_k=top_k)
            logger.info("Retrieved %s chunks for question", len(chunks))
            return chunks

