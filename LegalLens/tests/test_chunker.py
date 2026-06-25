from pathlib import Path

from chunking.chunker import ContractChunker
from utils.models import DocumentMetadata, PageText, ParsedDocument


def test_chunker_preserves_page_and_file_metadata() -> None:
    document = ParsedDocument(
        file_path=Path("contract.pdf"),
        metadata=DocumentMetadata(file_name="contract.pdf", page_count=1, document_length=200),
        pages=[PageText(file_name="contract.pdf", page_number=1, text="SECTION 1 NOTICE\n\nThe notice period is thirty days.")],
    )

    chunks = ContractChunker(chunk_size=80, chunk_overlap=10).chunk(document)

    assert chunks
    assert chunks[0].file_name == "contract.pdf"
    assert chunks[0].page_number == 1
    assert "notice period" in chunks[0].text.lower()


def test_chunker_assigns_nearest_heading() -> None:
    text = "SECTION 1 NOTICE\n\nThe notice period is thirty days.\n\nSECTION 2 PAYMENT\n\nInvoices are due monthly."
    document = ParsedDocument(
        file_path=Path("contract.pdf"),
        metadata=DocumentMetadata(file_name="contract.pdf", page_count=1, document_length=len(text)),
        pages=[PageText(file_name="contract.pdf", page_number=1, text=text)],
    )

    chunks = ContractChunker(chunk_size=40, chunk_overlap=0).chunk(document)

    assert any(chunk.section == "SECTION 1 NOTICE" for chunk in chunks)
    assert any(chunk.section == "SECTION 2 PAYMENT" for chunk in chunks)
