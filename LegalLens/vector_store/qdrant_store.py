import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models

from utils.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Manual Qdrant adapter for storing and retrieving contract chunks."""

    def __init__(self, url: str, collection_name: str, vector_size: int = 384) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(url=url)

    def ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        if any(collection.name == self.collection_name for collection in collections):
            info = self.client.get_collection(self.collection_name)
            vectors = info.config.params.vectors
            existing_size = getattr(vectors, "size", None)
            if existing_size and existing_size != self.vector_size:
                raise ValueError(
                    f"Qdrant collection {self.collection_name} has vector size {existing_size}, "
                    f"but LegalLens expected {self.vector_size}."
                )
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="file_name",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="page_number",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        logger.info("Created Qdrant collection: %s", self.collection_name)

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts must match.")
        self.ensure_collection()
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "file_name": chunk.file_name,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info("Upserted %s vectors into %s", len(points), self.collection_name)

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        self.ensure_collection()
        results = self.client.search(collection_name=self.collection_name, query_vector=query_vector, limit=top_k)
        retrieved: list[RetrievedChunk] = []
        for result in results:
            payload = result.payload or {}
            chunk = Chunk(
                chunk_id=str(payload["chunk_id"]),
                file_name=str(payload["file_name"]),
                page_number=int(payload["page_number"]),
                section=str(payload["section"]),
                text=str(payload["text"]),
                chunk_index=int(payload["chunk_index"]),
                token_count=int(payload.get("token_count", 0)),
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=float(result.score)))
        return retrieved

    def clear_collection(self) -> None:
        if any(collection.name == self.collection_name for collection in self.client.get_collections().collections):
            self.client.delete_collection(self.collection_name)
