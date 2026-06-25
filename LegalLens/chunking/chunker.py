import logging
from uuid import NAMESPACE_URL, uuid5

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logging import timed_step
from utils.models import Chunk, ParsedDocument
from utils.text import detect_heading, estimate_token_count

logger = logging.getLogger(__name__)


class ContractChunker:
    """Create citation-ready chunks while preserving page and section metadata."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
            length_function=lambda text: estimate_token_count(text),
        )

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        with timed_step(logger, f"Chunk generation: {document.metadata.file_name}"):
            chunks: list[Chunk] = []
            current_section = "General"

            for page in document.pages:
                section_blocks = self._section_blocks(page.text, current_section)
                if section_blocks:
                    current_section = section_blocks[-1][0]
                for section, section_text in section_blocks:
                    split_texts = self.splitter.split_text(section_text)
                    for split_text in split_texts:
                        assigned_section = self._nearest_section(split_text, section_text, section)
                        chunk_index = len(chunks)
                        chunk_id = self._chunk_id(document.metadata.file_name, page.page_number, chunk_index, split_text)
                        chunks.append(
                            Chunk(
                                chunk_id=chunk_id,
                                file_name=document.metadata.file_name,
                                page_number=page.page_number,
                                section=assigned_section,
                                text=split_text.strip(),
                                chunk_index=chunk_index,
                                token_count=estimate_token_count(split_text),
                            )
                        )
            logger.info("Generated %s chunks for %s", len(chunks), document.metadata.file_name)
            return chunks

    @staticmethod
    def _chunk_id(file_name: str, page_number: int, chunk_index: int, text: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"{file_name}:{page_number}:{chunk_index}:{text}"))

    @staticmethod
    def _nearest_section(chunk_text: str, page_text: str, fallback: str) -> str:
        chunk_start = page_text.find(chunk_text[: min(80, len(chunk_text))])
        headings: list[tuple[int, str]] = []
        cursor = 0
        for line in page_text.splitlines():
            position = page_text.find(line, cursor)
            cursor = position + len(line) if position >= 0 else cursor
            if detect_heading(line):
                headings.append((max(position, 0), line.strip()))

        if not headings:
            return fallback
        if chunk_start < 0:
            return headings[-1][1]

        nearest = fallback
        for position, heading in headings:
            if position <= chunk_start:
                nearest = heading
            else:
                break
        return nearest

    @staticmethod
    def _section_blocks(page_text: str, fallback: str) -> list[tuple[str, str]]:
        blocks: list[tuple[str, list[str]]] = []
        current_section = fallback
        current_lines: list[str] = []

        for line in page_text.splitlines():
            if detect_heading(line):
                if current_lines:
                    blocks.append((current_section, current_lines))
                current_section = line.strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            blocks.append((current_section, current_lines))
        return [(section, "\n".join(lines).strip()) for section, lines in blocks if "\n".join(lines).strip()]
