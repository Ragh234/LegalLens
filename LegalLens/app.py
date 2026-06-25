import streamlit as st

from config.settings import settings
from evaluation.evaluator import RAGEvaluator
from prompts.templates import build_prompt_parts
from rag_pipeline import LegalLensPipeline
from utils.logging import configure_logging
from utils.models import Chunk, RAGAnswer, RetrievedChunk
from utils.upload import UploadValidationError, save_validated_upload

configure_logging()


def get_pipeline() -> LegalLensPipeline:
    return LegalLensPipeline(settings)


def render_retrieval_inspector(chunks: list[RetrievedChunk]) -> None:
    st.subheader("Retrieval Inspector")
    if not chunks:
        st.info("No chunks retrieved.")
        return
    for index, item in enumerate(chunks, start=1):
        chunk = item.chunk
        with st.expander(f"{index}. {chunk.file_name} | page {chunk.page_number} | score {item.score:.4f}"):
            st.caption(f"Section: {chunk.section} | Chunk ID: {chunk.chunk_id} | Tokens: {chunk.token_count}")
            st.write(chunk.text)


def render_chunk_inspector(chunks: list[Chunk]) -> None:
    if not chunks:
        st.info("No chunks indexed in this session yet.")
        return
    for chunk in chunks:
        with st.expander(f"Chunk {chunk.chunk_index} | page {chunk.page_number} | {chunk.section}"):
            st.caption(f"Chunk ID: {chunk.chunk_id} | Size: {chunk.token_count} tokens | File: {chunk.file_name}")
            st.write(chunk.text)


def render_prompt_viewer(question: str, retrieved: list[RetrievedChunk]) -> None:
    prompt_parts = build_prompt_parts(question, retrieved)
    st.subheader("Prompt Viewer")
    with st.expander("System Prompt"):
        st.code(prompt_parts.system_prompt)
    with st.expander("Retrieved Context"):
        st.code(prompt_parts.retrieved_context)
    with st.expander("User Question"):
        st.code(prompt_parts.user_question)
    with st.expander("Exact Prompt Sent to Gemini"):
        st.code(prompt_parts.final_prompt)


def render_answer(answer: RAGAnswer) -> None:
    st.subheader("Generated Answer")
    if answer.error:
        st.warning(answer.error)
    st.markdown(answer.answer)
    st.caption(f"LLM latency: {answer.latency_ms:.2f} ms")

    st.subheader("Source Citations")
    if not answer.citations:
        st.info("No citations returned.")
        return
    for item in answer.citations:
        chunk = item.chunk
        st.markdown(
            f"- **{chunk.file_name}**, page **{chunk.page_number}**, section **{chunk.section}**, "
            f"chunk `{chunk.chunk_id}`, confidence score **{item.score:.4f}**"
        )


def main() -> None:
    st.set_page_config(page_title="LegalLens", layout="wide")
    st.title("LegalLens")
    st.caption("AI Contract Intelligence with an inspectable Retrieval-Augmented Generation pipeline")

    pipeline = get_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_chunks" not in st.session_state:
        st.session_state.indexed_chunks = []

    with st.sidebar:
        st.header("Pipeline Controls")
        uploaded_files = st.file_uploader("Upload PDF contracts", type=["pdf"], accept_multiple_files=True)
        top_k = st.slider("Top-K retrieval", min_value=1, max_value=12, value=settings.top_k)
        chunk_size = st.slider("Chunk size", min_value=300, max_value=1400, value=settings.chunk_size, step=50)
        chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=300, value=settings.chunk_overlap, step=25)

        if st.button("Ingest Contracts", use_container_width=True, disabled=not uploaded_files):
            with st.status("Processing contracts...", expanded=True) as status:
                total_chunks = 0
                for uploaded_file in uploaded_files or []:
                    try:
                        saved = save_validated_upload(uploaded_file)
                        if saved.duplicate:
                            st.write(f"Duplicate detected, reusing {saved.stored_name}")
                        else:
                            st.write(f"Saved {uploaded_file.name} as {saved.stored_name}")
                        st.write("Parsing, chunking, embedding, and storing vectors")
                        chunks = pipeline.ingest_contract(saved.path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        st.session_state.indexed_chunks.extend(chunks)
                        total_chunks += len(chunks)
                        st.write(f"Stored {len(chunks)} chunks from {uploaded_file.name}")
                    except UploadValidationError as exc:
                        st.warning(str(exc))
                    except Exception as exc:
                        st.error(f"Failed to ingest {uploaded_file.name}: {exc}")
                status.update(label=f"Ingestion complete: {total_chunks} chunks indexed", state="complete")

        st.divider()
        st.caption("Set `GEMINI_API_KEY` in `.env` before generating answers.")

    with st.expander("Chunk Inspector", expanded=False):
        render_chunk_inspector(st.session_state.indexed_chunks)

    question = st.chat_input("Ask a question about the uploaded contracts")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Retrieving relevant contract chunks..."):
            retrieved = pipeline.retrieve(question, top_k=top_k)

        render_retrieval_inspector(retrieved)
        render_prompt_viewer(question, retrieved)

        col_answer, col_eval = st.columns([2, 1])
        with col_answer:
            with st.spinner("Generating grounded answer with Gemini..."):
                answer = pipeline.answer(question, top_k=top_k, retrieved=retrieved)
            render_answer(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer.answer})

        with col_eval:
            evaluator = RAGEvaluator(pipeline.retriever)
            trace = evaluator.inspect(question, top_k, retrieved=retrieved)
            st.subheader("Evaluation Trace")
            st.metric("Trace latency", f"{trace.latency_ms:.2f} ms")
            st.metric("Retrieved chunks", len(trace.retrieved_chunks))
            with st.expander("Final Prompt"):
                st.code(trace.final_prompt)


if __name__ == "__main__":
    main()
