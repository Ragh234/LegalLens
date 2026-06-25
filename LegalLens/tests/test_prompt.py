from prompts.templates import build_prompt, build_prompt_parts
from utils.models import Chunk, RetrievedChunk


def test_prompt_contains_grounding_and_citations() -> None:
    chunk = Chunk(
        chunk_id="abc",
        file_name="msa.pdf",
        page_number=3,
        section="Termination",
        text="Either party may terminate with 30 days written notice.",
        chunk_index=0,
        token_count=10,
    )

    prompt = build_prompt("What is the notice period?", [RetrievedChunk(chunk=chunk, score=0.92)])

    assert "Answer only from the retrieved contract context" in prompt
    assert "msa.pdf" in prompt
    assert "30 days" in prompt
    assert "Similarity Score" in prompt


def test_prompt_parts_expose_system_context_and_question() -> None:
    parts = build_prompt_parts("Question?", [])

    assert "legal document assistant" in parts.system_prompt
    assert parts.user_question == "Question?"
    assert "User question" in parts.final_prompt
