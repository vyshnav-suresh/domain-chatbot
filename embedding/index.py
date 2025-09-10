from __future__ import annotations

import time
import logging
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer

log = logging.getLogger("embedding")


class VectorIndex:
    """
    Lightweight FAISS index wrapper storing embeddings and metadata in-memory.
    Not persisted to disk; intended to live in app session state.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadatas: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def _ensure_model(self):
        if self.model is None:
            t0 = time.perf_counter()
            self.model = SentenceTransformer(self.model_name)
            log.info("[MODEL] Loaded SentenceTransformer '%s' in %.2fs", self.model_name, time.perf_counter() - t0)

    def _ensure_index(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
            log.info("[FAISS] Created IndexFlatIP(dim=%d)", dim)

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        # Ensure 2D shape
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        t0 = time.perf_counter()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        out = vecs / norms
        log.debug("[NORM] Normalized vectors shape=%s in %.3fs", out.shape, time.perf_counter() - t0)
        return out

    def build(self, chunks: List[Dict[str, Any]]):
        """
        Build index from scratch with provided chunks.
        Each chunk dict must contain a 'text' field and metadata.
        """
        self._ensure_model()
        if not chunks:
            # Nothing to build
            self.index = None
            self.embeddings = None
            self.metadatas = []
            return
        texts = [c["text"] for c in chunks]
        if not texts:
            self.index = None
            self.embeddings = None
            self.metadatas = []
            return
        t0 = time.perf_counter()
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        log.info("[EMBED] Encoded %d chunk(s) in %.2fs", len(texts), time.perf_counter() - t0)
        embs = np.asarray(embs, dtype="float32")
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        embs = self._normalize(embs)
        dim = embs.shape[1]
        self._ensure_index(dim)
        # reset index
        self.index = faiss.IndexFlatIP(dim)
        t1 = time.perf_counter()
        self.index.add(embs)
        log.info("[INDEX] Added %d vectors to new index in %.2fs (dim=%d)", embs.shape[0], time.perf_counter() - t1, dim)
        self.embeddings = embs
        self.metadatas = chunks

    def add(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
        self._ensure_model()
        texts = [c["text"] for c in chunks]
        if not texts:
            return
        t0 = time.perf_counter()
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        log.info("[EMBED] Encoded %d new chunk(s) in %.2fs", len(texts), time.perf_counter() - t0)
        embs = np.asarray(embs, dtype="float32")
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        embs = self._normalize(embs)
        if self.index is None:
            self._ensure_index(embs.shape[1])
            self.embeddings = embs
            t1 = time.perf_counter()
            self.index.add(embs)
            log.info("[INDEX] Added %d vectors to new index in %.2fs", embs.shape[0], time.perf_counter() - t1)
            self.metadatas = list(chunks)
        else:
            t1 = time.perf_counter()
            self.index.add(embs)
            log.info("[INDEX] Appended %d vectors to existing index in %.2fs", embs.shape[0], time.perf_counter() - t1)
            if self.embeddings is None:
                self.embeddings = embs
            else:
                self.embeddings = np.vstack([self.embeddings, embs])
            self.metadatas.extend(chunks)

    def search(self, query: str, top_k: int = 8) -> List[Tuple[Dict[str, Any], float]]:
        if self.index is None or self.model is None:
            return []
        t0 = time.perf_counter()
        q_emb = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        q_emb = self._normalize(q_emb.astype("float32"))
        scores, idxs = self.index.search(q_emb, top_k)
        log.info("[SEARCH] Query top_k=%d completed in %.2fs", top_k, time.perf_counter() - t0)
        scores = scores[0]
        idxs = idxs[0]
        results: List[Tuple[Dict[str, Any], float]] = []
        for i, s in zip(idxs, scores):
            if i == -1:
                continue
            meta = self.metadatas[i]
            results.append((meta, float(s)))
        return results
