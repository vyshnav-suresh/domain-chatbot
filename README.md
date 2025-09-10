# Domain-Specific RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for answering user queries from domain-specific documents, with citations and uncertainty handling.

## Project Structure

- `ingestion/` — Document parsing and chunking
- `embedding/` — Embedding and vector indexing
- `retrieval/` — Hybrid retrieval and reranking
- `generation/` — LLM answer generation, citation, and uncertainty
- `ui/` — Streamlit web interface
- `evaluation/` — Evaluation scripts and test data
- `main.py` — Entrypoint for the Streamlit app
- `requirements.txt` — Python dependencies
- `README.md` — Setup and usage instructions

## Quick Start

1. Install requirements: `pip install -r requirements.txt`
2. Set LLM API key (Gemini) in .env:
   - GOOGLE_API_KEY = "<your_key>"`
3. Run the app: `streamlit run main.py`

## Features
- Upload PDF/text docs
- Chunking with overlap and metadata
- Embedding (Sentence Transformers/OpenAI)
- Local FAISS vector store
- Hybrid retrieval + reranking
- LLM-based answer generation with citations (Gemini by default)
- Uncertainty and fallback handling
- Minimal Streamlit UI
- Evaluation framework

## Requirements
- Python 3.10+
- CPU/GPU for embeddings (optional)

---

## Architecture Overview

The app implements a classic RAG pipeline:

- __Ingestion & Chunking__ (`ingestion/ingest.py`)
  - PDFs are read page-by-page, then split into ~500-token chunks with a 50‑token overlap.
  - TXT files are read fully, then chunked similarly.
  - Each chunk is annotated with:
    - `document` (file name)
    - `page` (for PDFs, 1-based)
    - `chunk_index` (index within that page’s chunks)
    - `section_id` (best-effort detection like 1.1, 2.3)
    - `section_hint` (first ~12 words of the chunk)

- __Embeddings & Vector DB__ (`embedding/index.py`)
  - Model: `sentence-transformers` — `all-MiniLM-L6-v2` by default.
  - Vectors are L2-normalized; stored in __FAISS__ `IndexFlatIP`.
  - Because vectors are unit-length, inner product ≡ cosine similarity.
  - The index and metadata are kept in memory (Streamlit session); not persisted to disk.

- __Retrieval & Reranking__ (`retrieval/retriever.py`)
  - Step 1: Vector search (top‑K, cosine similarity from FAISS).
  - Step 2: Threshold filter (default 0.7). If none pass, keep top 3.
  - Step 3: Cross‑encoder reranking `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - Final score = `0.4 * normalized_vec_sim + 0.6 * rerank_score`.

- __Answer Generation & Streaming__ (`generation/generator.py`)
  - If a Gemini API key exists, the app prompts `gemini-1.5-flash` to answer using only the retrieved contexts.
  - Inline citations are requested as `[Context N]`, matching the context numbering.
  - Sources are shown as: `[Context N] (score=…) [page X,sec] DocumentName — short hint`.
  - A safe fallback message is used when the model is unavailable or retrieval is weak.

- __UI__ (`main.py`)
  - Streamlit app with document upload, compact suggestion chips, chat panel, and a sources expander.
  - Suggestions auto‑generate after indexing and can be toggled from the sidebar.

## Ingestion & Chunking Details

- __Token-aware splitting__: Uses LangChain `TokenTextSplitter` with `cl100k_base` when available.
- __Parameters__: `CHUNK_SIZE=500`, `CHUNK_OVERLAP=50` (in `ingestion/ingest.py`).
- __Fallback__: Word‑based sliding window if the token splitter is unavailable.

## Vector Database (FAISS)

- Backend: `faiss.IndexFlatIP` (inner product).
- Normalization: All vectors are L2‑normalized; thus IP ≡ cosine similarity.
- Persistence: In‑memory only (Streamlit session). If you need persistence, we can extend to save/load the index and metadata.

## Retriever Pipeline

- Initial recall from FAISS (top‑K; default configurable via `retrieval/retriever.py`).
- Filter by similarity threshold; keep top 3 if none pass to avoid empty answers.
- Rerank with CrossEncoder on `(query, chunk_text)` pairs.
- Combine scores and return the top‑N chunks with metadata and scores.

## Running the App

1. `pip install -r requirements.txt`
2. Create `.env` with your Gemini key:
   ```env
   GOOGLE_API_KEY="<your_key>"
   ```
3. `streamlit run main.py`
4. Upload PDF/TXT files in the sidebar.
5. Ask questions in the chat box (bottom). Open “Sources” to see page/section‑aware citations.

## Evaluation

- Script: `evaluation/run_eval.py`
  - Inputs: a docs directory and a JSONL dataset (`evaluation/dataset.jsonl` or your own).
  - Outputs: a JSON report with:
    - __accuracy__: simple string‑match vs. `expected_answer`
    - __source_correctness__: fraction of `expected_sources` found in citations
    - __avg_latency_s__, __fallback_rate__
    - per‑item details (scores, citations, latency)
- Helper: `evaluation/prepare_eval.py` can generate 15–20 candidate questions from your docs for you to label.

## Logging

- Structured logs are emitted to the terminal with timings for ingestion, indexing, retrieval, generation, and session events.
- Look for tags like `[INGEST]`, `[INDEX]`, `[RETRIEVE]`, `[GEN]`, `[SUGGEST]`, `[SESSION]`.

## Troubleshooting

- __No answers / empty sources__:
  - Ensure documents were successfully ingested (check terminal logs).
  - Lower the retrieval `similarity_threshold` or increase `top_k`.
- __Slow retrieval__:
  - First query will load models; subsequent queries are faster.
  - Consider disabling reranking or reducing `rerank_top_n`.
- __Model/LLM errors__:
  - Verify `GOOGLE_API_KEY` in `.env`.
  - The app will fall back to a safe extractive answer if generation fails.

## Configuration Cheatsheet

- `ingestion/ingest.py`: `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `embedding/index.py`: embedding model (`all-MiniLM-L6-v2`)
- `retrieval/retriever.py`: `DEFAULT_TOP_K`, `similarity_threshold`, reranker model
- `generation/generator.py`: prompt style, model name, citation format

---

If you need persistence (saving/loading FAISS) or want to swap FAISS for a hosted vector DB (Chroma, Qdrant, Pinecone, Milvus), open an issue or PR—`VectorIndex` was designed as a thin adapter and can be extended.
