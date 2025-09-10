from __future__ import annotations

import time
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder

from embedding.index import VectorIndex

log = logging.getLogger("retrieval")

DEFAULT_TOP_K = 5
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Retriever:
    def __init__(self, index: VectorIndex):
        self.index = index
        self._reranker: CrossEncoder | None = None

    def _ensure_reranker(self):
        if self._reranker is None:
            t0 = time.perf_counter()
            self._reranker = CrossEncoder(RERANK_MODEL)
            log.info("[RERANK] Loaded CrossEncoder '%s' in %.2fs", RERANK_MODEL, time.perf_counter() - t0)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        rerank_top_n: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[Tuple[Dict[str, Any], float]]:
        t_total = time.perf_counter()
        log.info("[RETRIEVE] query='%s' top_k=%d threshold=%.2f", (query[:80] + "…") if len(query) > 80 else query, top_k, similarity_threshold)
        # Step 1: initial semantic search
        t0 = time.perf_counter()
        initial = self.index.search(query, top_k=top_k)
        log.info("[RETRIEVE] Initial search returned %d item(s) in %.2fs", len(initial), time.perf_counter() - t0)
        if not initial:
            return []

        # Filter by similarity threshold first-pass
        filtered = [(m, s) for (m, s) in initial if s >= similarity_threshold]
        log.info("[RETRIEVE] Filtered to %d item(s) by threshold", len(filtered))
        if not filtered:
            # if none pass threshold, keep top 3 anyway to allow partial answers
            filtered = initial[:3]
            log.info("[RETRIEVE] None passed threshold; kept %d top item(s)", len(filtered))

        # Step 2: reranking
        self._ensure_reranker()
        pairs = [(query, m["text"]) for (m, _s) in filtered]
        t1 = time.perf_counter()
        scores = self._reranker.predict(pairs)
        log.info("[RERANK] Predicted scores for %d pairs in %.2fs", len(pairs), time.perf_counter() - t1)
        # Create a combined score: average normalized vector sim and reranker score
        # Vector sims are already cosine similarities in [-1, 1]; reranker outputs relevance ~ [0, 1]
        # Normalize vector sim to [0,1]
        vec_sims = np.array([s for (_m, s) in filtered])
        vec_sims_norm = (vec_sims + 1.0) / 2.0
        rerank_scores = np.array(scores)
        combined = 0.4 * vec_sims_norm + 0.6 * rerank_scores

        items = [filtered[i][0] for i in range(len(filtered))]
        ranked_idx = np.argsort(-combined)
        reranked = [(items[i], float(combined[i])) for i in ranked_idx[:rerank_top_n]]
        log.info("[RETRIEVE] Finished in %.2fs; returning %d reranked item(s)", time.perf_counter() - t_total, len(reranked))
        return reranked


def retrieve_chunks(query: str, index: VectorIndex, top_k: int = DEFAULT_TOP_K, rerank_top_n: int = 5, similarity_threshold: float = 0.7):
    retriever = Retriever(index)
    return retriever.retrieve(query, top_k=top_k, rerank_top_n=rerank_top_n, similarity_threshold=similarity_threshold)
