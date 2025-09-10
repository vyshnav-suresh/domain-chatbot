import os
import time
import re
import logging
from typing import List, Dict, Any, Tuple
from PyPDF2 import PdfReader
import pdfplumber
from pathlib import Path
import tiktoken
try:
    # LangChain v0.3+ splitters
    from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter  # type: ignore
except Exception:
    try:
        # Older LangChain
        from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter  # type: ignore
    except Exception:
        TokenTextSplitter = None  # type: ignore
        RecursiveCharacterTextSplitter = None  # type: ignore

CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens

# Module logger
log = logging.getLogger("ingestion")

try:
    encoding = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text):
        return len(encoding.encode(text))
except ImportError:
    def count_tokens(text):
        return len(text.split())


def extract_text_from_pdf(file) -> str:
    try:
        t0 = time.perf_counter()
        pdf = PdfReader(file)
        text = "\n".join(page.extract_text() or '' for page in pdf.pages)
        log.info("[PDF] Extracted %d chars from %d page(s) in %.2fs", len(text), len(pdf.pages), time.perf_counter() - t0)
        return text
    except Exception:
        try:
            t1 = time.perf_counter()
            with pdfplumber.open(file) as pdf:
                text = "\n".join(page.extract_text() or '' for page in pdf.pages)
                log.info("[PDF:plumber] Extracted %d chars from %d page(s) in %.2fs", len(text), len(pdf.pages), time.perf_counter() - t1)
                return text
        except Exception as e:
            raise RuntimeError(f"PDF parsing failed: {e}")


def extract_pages_from_pdf(file) -> List[Dict[str, Any]]:
    """Return a list of pages with their text and 1-based page numbers.
    Each item: {"page": int, "text": str}
    """
    pages: List[Dict[str, Any]] = []
    try:
        t0 = time.perf_counter()
        pdf = PdfReader(file)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ''
            pages.append({"page": i + 1, "text": text})
        log.info("[PDF] Extracted %d page(s) via PyPDF2 in %.2fs", len(pages), time.perf_counter() - t0)
        return pages
    except Exception:
        try:
            t1 = time.perf_counter()
            with pdfplumber.open(file) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ''
                    pages.append({"page": i + 1, "text": text})
            log.info("[PDF:plumber] Extracted %d page(s) in %.2fs", len(pages), time.perf_counter() - t1)
            return pages
        except Exception as e:
            raise RuntimeError(f"PDF page parsing failed: {e}")


def extract_text_from_txt(file) -> str:
    t0 = time.perf_counter()
    data = file.read()
    if isinstance(data, bytes):
        text = data.decode("utf-8", errors="ignore")
    else:
        text = str(data)
    log.info("[TXT] Read %d chars in %.2fs", len(text), time.perf_counter() - t0)
    return text


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Chunk text with a preference for LangChain's TokenTextSplitter (token-based, tiktoken-aware).
    Falls back to simple word-based sliding window if splitters are unavailable.
    """
    t0 = time.perf_counter()
    chunks: List[Dict[str, Any]] = []
    # Preferred: LangChain token-based splitting
    if TokenTextSplitter is not None:
        try:
            splitter = TokenTextSplitter(
                encoding_name="cl100k_base",
                chunk_size=chunk_size,
                chunk_overlap=overlap,
            )
            parts = splitter.split_text(text)
            for i, p in enumerate(parts):
                chunks.append({
                    "chunk_id": i,
                    "text": p,
                })
            log.info("[CHUNK] Token splitter produced %d chunk(s) in %.2fs", len(chunks), time.perf_counter() - t0)
            return chunks
        except Exception:
            pass
    # Fallback: word-based sliding window
    words = text.split()
    start = 0
    chunk_id = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_val = " ".join(chunk_words)
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text_val,
            "start_word": start,
            "end_word": end
        })
        chunk_id += 1
        start += max(1, chunk_size - overlap)
    return chunks


def ingest_documents(files) -> Dict[str, Any]:
    """
    Ingests and chunks uploaded files. Returns stats and saves chunk metadata for embedding.
    """
    t0 = time.perf_counter()
    all_chunks = []
    doc_names = []
    for file in files:
        name = getattr(file, 'name', 'uploaded')
        doc_names.append(name)
        ext = os.path.splitext(name)[-1].lower()
        if ext == ".pdf":
            log.info("[INGEST] Processing PDF: %s", name)
            # Extract per-page and chunk within each page to preserve page refs
            pages = extract_pages_from_pdf(file)
            for pg in pages:
                page_no = pg.get("page", None)
                text = (pg.get("text", "") or "").strip()
                if not text:
                    continue
                page_chunks = chunk_text(text)
                # Try to detect a section identifier like '1.1' from the page text
                sec_match = re.search(r"(?m)^\s*(\d+(?:\.\d+)+)\b", text)
                page_section_id = sec_match.group(1) if sec_match else None
                for i, chunk in enumerate(page_chunks):
                    chunk["document"] = name
                    chunk["chunk_index"] = i
                    if page_no is not None:
                        chunk["page"] = page_no
                    if page_section_id:
                        chunk["section_id"] = page_section_id
                    # add a small section hint from the start of the chunk
                    ch_text = chunk.get("text", "").strip()
                    if ch_text:
                        hint = " ".join(ch_text.split()[:12])
                        chunk["section_hint"] = hint
                # Filter empties and collect
                non_empty = [c for c in page_chunks if c.get("text", "").strip()]
                log.info("[INGEST] %s p.%s => %d/%d non-empty chunk(s)", name, page_no, len(non_empty), len(page_chunks))
                all_chunks.extend(non_empty)
            # Done with this file; continue to next
            continue
        elif ext == ".txt":
            log.info("[INGEST] Processing TXT: %s", name)
            text = extract_text_from_txt(file)
        else:
            log.warning("[INGEST] Skipping unsupported file type: %s", name)
            continue  # skip unsupported
        text = (text or "").strip()
        if not text:
            log.warning("[INGEST] Empty text extracted; skipping: %s", name)
            continue
        chunks = chunk_text(text)
        # Try to detect a section identifier like '1.1' from the document text
        sec_match = re.search(r"(?m)^\s*(\d+(?:\.\d+)+)\b", text)
        doc_section_id = sec_match.group(1) if sec_match else None
        for i, chunk in enumerate(chunks):
            chunk["document"] = name
            chunk["chunk_index"] = i
            if doc_section_id:
                chunk["section_id"] = doc_section_id
            # add a small section hint from the start of the chunk
            ch_text = chunk.get("text", "").strip()
            if ch_text:
                hint = " ".join(ch_text.split()[:12])
                chunk["section_hint"] = hint
        # Filter out empty chunk texts
        non_empty = [c for c in chunks if c.get("text", "").strip()]
        log.info("[INGEST] %s => %d/%d non-empty chunk(s)", name, len(non_empty), len(chunks))
        all_chunks.extend(non_empty)
    # Optionally: Save chunks to disk or pass to embedding pipeline
    dur = time.perf_counter() - t0
    log.info("[INGEST] Completed ingest_documents in %.2fs | docs=%d chunks=%d", dur, len(doc_names), len(all_chunks))
    return {"num_docs": len(doc_names), "num_chunks": len(all_chunks), "chunks": all_chunks}


def ingest_dir(directory: str) -> Dict[str, Any]:
    """
    Ingest all PDF/TXT files from a directory path. Used by the evaluation script.
    """
    p = Path(directory)
    all_files = sorted([*p.glob("**/*.pdf"), *p.glob("**/*.txt")])
    # Mimic Streamlit uploader objects minimally: supply .name and .read interface
    class _FSFile:
        def __init__(self, path: Path):
            self.path = path
            self.name = path.name
        def read(self):
            return self.path.read_bytes()
    fs_files = [_FSFile(fp) for fp in all_files]
    return ingest_documents(fs_files)
