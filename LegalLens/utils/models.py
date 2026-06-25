from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PageText:
    file_name: str
    page_number: int
    text: str
    headings: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentMetadata:
    file_name: str
    page_count: int
    document_length: int


@dataclass(frozen=True)
class ParsedDocument:
    file_path: Path
    metadata: DocumentMetadata
    pages: list[PageText]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file_name: str
    page_number: int
    section: str
    text: str
    chunk_index: int
    token_count: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    citations: list[RetrievedChunk]
    prompt: str
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True)
class PromptParts:
    system_prompt: str
    retrieved_context: str
    user_question: str
    final_prompt: str


@dataclass(frozen=True)
class UploadedContract:
    original_name: str
    stored_name: str
    path: Path
    size_bytes: int
    duplicate: bool = False
