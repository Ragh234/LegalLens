import logging
from pathlib import Path

import fitz
import pdfplumber

from utils.logging import log_event, timed_step
from utils.models import DocumentMetadata, PageText, ParsedDocument
from utils.text import detect_heading, normalize_whitespace, remove_repeated_headers_footers

logger = logging.getLogger(__name__)


class PDFParser:
    """Extract page-aware text, headings, and tables from legal PDF contracts."""

    def parse(self, file_path: Path) -> ParsedDocument:
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported file type: {file_path.suffix}. LegalLens currently supports PDF files.")
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        with timed_step(logger, f"Document parsing: {file_path.name}"):
            pages = self._extract_pages(file_path)
            document_length = sum(len(page.text) for page in pages)
            metadata = DocumentMetadata(
                file_name=file_path.name,
                page_count=len(pages),
                document_length=document_length,
            )
            return ParsedDocument(file_path=file_path, metadata=metadata, pages=pages)

    def _extract_pages(self, file_path: Path) -> list[PageText]:
        pages: list[PageText] = []
        table_map = self._extract_tables(file_path)

        with fitz.open(file_path) as document:
            for index, page in enumerate(document, start=1):
                text = page.get_text("text")
                text = remove_repeated_headers_footers(text)
                text = normalize_whitespace(text)
                tables = table_map.get(index, [])
                if tables:
                    text = f"{text}\n\n" + "\n\n".join(tables)
                if not _has_meaningful_text(text):
                    log_event(logger, "empty_page_skipped", file_name=file_path.name, page_number=index)
                    continue
                headings = [line.strip() for line in text.splitlines() if detect_heading(line)]
                pages.append(
                    PageText(
                        file_name=file_path.name,
                        page_number=index,
                        text=text,
                        headings=headings,
                        tables=tables,
                    )
                )
        return pages

    def _extract_tables(self, file_path: Path) -> dict[int, list[str]]:
        tables_by_page: dict[int, list[str]] = {}
        try:
            with pdfplumber.open(file_path) as pdf:
                for index, page in enumerate(pdf.pages, start=1):
                    page_tables: list[str] = []
                    for table in page.extract_tables() or []:
                        rows = [" | ".join(cell or "" for cell in row) for row in table if row]
                        if rows:
                            page_tables.append("\n".join(rows))
                    if page_tables:
                        tables_by_page[index] = page_tables
        except Exception as exc:
            logger.warning("Table extraction failed for %s: %s", file_path.name, exc)
        return tables_by_page


def _has_meaningful_text(text: str) -> bool:
    normalized = normalize_whitespace(text)
    alnum_count = sum(char.isalnum() for char in normalized)
    return alnum_count >= 20
