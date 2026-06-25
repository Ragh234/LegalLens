from utils.models import PromptParts, RetrievedChunk

SYSTEM_PROMPT = """You are a legal document assistant.
Answer only from the retrieved contract context.
Never invent facts, clauses, dates, obligations, parties, or legal interpretations that are not present in the context.
If the answer is unavailable, say: "I could not find that information in the uploaded contracts."
Always cite the source using document name, page number, section, and chunk ID.
This is educational assistance, not legal advice."""


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for item in chunks:
        chunk = item.chunk
        blocks.append(
            "\n".join(
                [
                    f"[Document: {chunk.file_name}]",
                    f"[Page: {chunk.page_number}]",
                    f"[Section: {chunk.section}]",
                    f"[Chunk ID: {chunk.chunk_id}]",
                    f"[Similarity Score: {item.score:.4f}]",
                    chunk.text,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def build_prompt_parts(question: str, chunks: list[RetrievedChunk]) -> PromptParts:
    context = build_context(chunks)
    final_prompt = f"""{SYSTEM_PROMPT}

Retrieved contract context:
{context}

User question:
{question}

Return a concise answer followed by a "Citations" section. Include confidence based on how directly the retrieved context answers the question."""
    return PromptParts(
        system_prompt=SYSTEM_PROMPT,
        retrieved_context=context,
        user_question=question,
        final_prompt=final_prompt,
    )


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return build_prompt_parts(question, chunks).final_prompt
