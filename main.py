# Main entry point for the Streamlit RAG chatbot

import logging
import time
import streamlit as st
from dotenv import load_dotenv
from ingestion.ingest import ingest_documents
from embedding.index import VectorIndex
from retrieval.retriever import retrieve_chunks
from generation.generator import generate_answer, stream_answer, suggest_questions

load_dotenv()  # Load environment variables from .env if present

# Configure logging (console)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")

st.set_page_config(page_title="Domain RAG Chatbot", layout="wide")
# Global compact CSS for suggestion chips
st.markdown(
    """
    <style>
    .compact-btn button {
        padding: 0.25rem 0.6rem !important;
        font-size: 0.85rem !important;
        border-radius: 999px !important;
        line-height: 1.2 !important;
        white-space: nowrap !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("📄 Domain-Specific RAG Chatbot")
st.caption("Step 1: Upload documents in the sidebar • Step 2: Ask a question below")

# Sidebar: Document upload and stats
st.sidebar.header("Upload Documents")
doc_files = st.sidebar.file_uploader("Choose PDF or text files", type=["pdf", "txt"], accept_multiple_files=True)

if doc_files:
    log.info("[UI] Received %d file(s) for ingestion", len(doc_files))
    with st.spinner("Ingesting and indexing documents..."):
        t0 = time.perf_counter()
        stats = ingest_documents(doc_files)
        log.info("[INGEST] Completed ingestion in %.2fs | docs=%s chunks=%s", time.perf_counter() - t0, stats.get("num_docs"), stats.get("num_chunks"))
        if stats.get("chunks"):
            # Build a fresh vector index in session state
            st.session_state["vector_index"] = VectorIndex()
            t1 = time.perf_counter()
            st.session_state["vector_index"].build(stats["chunks"])
            log.info("[INDEX] Built vector index in %.2fs", time.perf_counter() - t1)
            st.session_state["doc_stats"] = {"num_docs": stats["num_docs"], "num_chunks": stats["num_chunks"]}
            # Generate fresh suggestions immediately after indexing
            t2 = time.perf_counter()
            st.session_state["suggestions"] = suggest_questions(getattr(st.session_state["vector_index"], "metadatas", []), min_q=5, max_q=10) or []
            log.info("[SUGGESTIONS] Generated %d suggestion(s) in %.2fs", len(st.session_state["suggestions"]), time.perf_counter() - t2)
            st.sidebar.success(f"Indexed {stats['num_chunks']} chunks from {stats['num_docs']} documents.")
        else:
            st.sidebar.warning("No text content found in uploaded files. Please try different documents.")
else:
    st.sidebar.info("Please upload at least one document.")

# Sidebar: UI preferences
st.sidebar.markdown("---")
st.sidebar.subheader("Preferences")
show_suggestions = st.sidebar.checkbox("Show suggested questions", value=True)

# Chat state
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # [{role: "user"|"assistant", content: str, citations?: List[str]}]
if "busy" not in st.session_state:
    st.session_state["busy"] = False

# Empty state guidance when no index is ready
if "vector_index" not in st.session_state:
    with st.container():
        st.info("No documents indexed yet. Upload PDFs or TXT files from the sidebar to enable retrieval-augmented answers.")
        st.markdown("- Supported types: `pdf`, `txt`\n- Larger files may take longer to process\n- After indexing, try the suggested questions that appear here")

# Render existing chat history
for msg in st.session_state["messages"]:
    with st.chat_message("user" if msg.get("role") == "user" else "assistant"):
        st.markdown(msg.get("content", ""))
        if msg.get("role") == "assistant" and msg.get("citations"):
            with st.expander("Sources", expanded=False):
                for src in msg["citations"]:
                    st.markdown(f"- {src}")

selected = None
if show_suggestions and "vector_index" in st.session_state and st.session_state.get("doc_stats", {}).get("num_chunks", 0) > 0:
    idx = st.session_state["vector_index"]
    # Generate or refresh suggestions
    if "suggestions" not in st.session_state:
        st.session_state["suggestions"] = suggest_questions(getattr(idx, "metadatas", []), min_q=5, max_q=10) or []
    with st.container():
        with st.expander("Suggested questions", expanded=False):
            raw_suggestions = st.session_state.get("suggestions", [])
            # Compact: limit count and shorten labels
            def _shorten(text: str, max_len: int = 60) -> str:
                t = text.strip().strip('"')
                if len(t) <= max_len:
                    return t
                return t[: max_len - 1].rstrip(" ,.;:") + "…"
            suggestions = [_shorten(s) for s in raw_suggestions[:6]]
            if suggestions:
                cols = st.columns(min(6, len(suggestions)))
                for i, s in enumerate(suggestions):
                    with cols[i % len(cols)]:
                        st.markdown('<div class="compact-btn">', unsafe_allow_html=True)
                        if st.button(s, key=f"sugg_{i}") and not st.session_state.get("busy", False):
                            selected = s
                        st.markdown('</div>', unsafe_allow_html=True)

# Chat input (always render at the bottom; disabled until index is ready)
chat_prompt = st.chat_input(
    "Ask about your documents...",
    disabled=(st.session_state.get("busy", False) or ("vector_index" not in st.session_state))
)
prompt = selected or chat_prompt
if prompt:
    log.info("[CHAT] User prompt received: %s", (prompt[:120] + "…") if len(prompt) > 120 else prompt)
    # Append user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vector_index" not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("Please upload documents first to build the index.")
    else:
        # Retrieve relevant chunks
        idx = st.session_state["vector_index"]
        collected = ""
        citations = []
        try:
            st.session_state["busy"] = True
            tr = time.perf_counter()
            retrieved = retrieve_chunks(prompt, index=idx)
            log.info("[RETRIEVE] Retrieved %d chunk(s) in %.2fs", len(retrieved), time.perf_counter() - tr)
            # Stream assistant response
            with st.chat_message("assistant"):
                tg = time.perf_counter()
                answer_stream, citations = stream_answer(prompt, retrieved)
                log.info("[GENERATE] Starting stream; citations=%d", len(citations))
                container = st.empty()
                for piece in answer_stream:
                    collected += piece
                    container.markdown(collected)
                # After stream completes, show sources below
                log.info("[GENERATE] Completed streaming in %.2fs (len=%d)", time.perf_counter() - tg, len(collected))
                if citations:
                    with st.expander("Sources", expanded=False):
                        for src in citations:
                            st.markdown(f"- {src}")
        except Exception as e:
            # Show error but keep chat area intact
            with st.chat_message("assistant"):
                st.error(f"An error occurred while generating the answer: {e}")
            log.exception("[ERROR] Exception during retrieve/generate: %s", e)
        finally:
            # Save assistant message (even if empty/error, to preserve flow)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": collected if collected else "",
                "citations": citations,
            })
            st.session_state["busy"] = False
            log.info("[CHAT] Assistant message saved; busy=False")

# Stats panel
if "doc_stats" in st.session_state:
    stats = st.session_state["doc_stats"]
    st.sidebar.markdown("---")
    st.sidebar.subheader("Index Stats")
    st.sidebar.text(f"Documents: {stats['num_docs']}")
    st.sidebar.text(f"Chunks: {stats['num_chunks']}")

# Sidebar session actions
st.sidebar.markdown("---")
st.sidebar.subheader("Session")
if st.sidebar.button("Clear Chat", help="Reset the conversation and suggestions"):
    st.session_state["messages"] = []
    # Keep suggestions if index exists; regenerate otherwise on next render
    if "vector_index" in st.session_state:
        st.session_state["suggestions"] = suggest_questions(getattr(st.session_state["vector_index"], "metadatas", []), min_q=5, max_q=10) or []
        log.info("[SESSION] Cleared chat; regenerated %d suggestion(s)", len(st.session_state["suggestions"]))

if __name__ == "__main__":
    pass
