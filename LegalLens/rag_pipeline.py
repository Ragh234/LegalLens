import logging
from pathlib import Path
from time import perf_counter

from chunking.chunker import ContractChunker
from config.settings import Settings
from embeddings.embedder import BGEEmbedder
from llm.gemini_client import GeminiClient
from parser.pdf_parser import PDFParser
from prompts.templates import build_prompt_parts
from retrieval.retriever import SemanticRetriever
from utils.logging import log_event, timed_step
from utils.models import Chunk, RAGAnswer, RetrievedChunk
from vector_store.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class LegalLensPipeline:
    """End-to-end RAG pipeline facade used by Streamlit and tests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.parser = PDFParser()
        self.chunker = ContractChunker(settings.chunk_size, settings.chunk_overlap)
        self.embedder = BGEEmbedder(settings.embedding_model)
        self.vector_store = QdrantVectorStore(settings.qdrant_url, settings.qdrant_collection, settings.vector_size)
        self.retriever = SemanticRetriever(self.embedder, self.vector_store)

    def ingest_contract(self, file_path: Path, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[Chunk]:
        start = perf_counter()
        parser = self.parser
        chunker = ContractChunker(chunk_size or self.settings.chunk_size, chunk_overlap or self.settings.chunk_overlap)

        with timed_step(logger, f"Total ingestion pipeline: {file_path.name}"):
            parsed = parser.parse(file_path)
            chunks = chunker.chunk(parsed)
            vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
            self.vector_store.upsert_chunks(chunks, vectors)

        logger.info("Ingested %s in %.2f ms", file_path.name, (perf_counter() - start) * 1000)
        return chunks

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        return self.retriever.retrieve(question, top_k)

    def answer(self, question: str, top_k: int, retrieved: list[RetrievedChunk] | None = None) -> RAGAnswer:
        start = perf_counter()
        try:
            retrieved = retrieved if retrieved is not None else self.retrieve(question, top_k)
            if not retrieved:
                latency_ms = (perf_counter() - start) * 1000
                log_event(logger, "answer_pipeline_completed", latency_ms=round(latency_ms, 2), retrieved_chunks=0)
                return RAGAnswer(
                    answer="I could not find that information in the uploaded contracts.",
                    citations=[],
                    prompt="",
                    latency_ms=latency_ms,
                )

            with timed_step(logger, "Prompt construction"):
                prompt = build_prompt_parts(question, retrieved).final_prompt
            client = GeminiClient(
                api_key=self.settings.gemini_api_key,
                model_name=self.settings.llm_model,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                timeout_seconds=self.settings.llm_timeout_seconds,
                max_retries=self.settings.llm_max_retries,
            )
            answer = client.generate(prompt, retrieved)
            log_event(
                logger,
                "answer_pipeline_completed",
                latency_ms=round((perf_counter() - start) * 1000, 2),
                retrieved_chunks=len(retrieved),
                llm_error=bool(answer.error),
            )
            return answer
        except ValueError:
            raise
