import logging
import os
from collections.abc import Sequence

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from sentence_transformers import SentenceTransformer

from utils.logging import timed_step

logger = logging.getLogger(__name__)


class BGEEmbedder:
    """Thin wrapper around BAAI/bge-small-en-v1.5 for query and chunk embeddings."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        with timed_step(logger, f"Embedding generation for {len(texts)} chunks"):
            vectors = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
            return [vector.tolist() for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        with timed_step(logger, "Query embedding"):
            vector = self.model.encode(query, normalize_embeddings=True, show_progress_bar=False)
            return vector.tolist()
