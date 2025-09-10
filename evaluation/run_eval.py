from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

from ingestion.ingest import ingest_dir
from embedding.index import VectorIndex
from retrieval.retriever import retrieve_chunks
from generation.generator import generate_answer


def load_dataset(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def contains_expected(answer: str, expected: str) -> bool:
    if not expected or expected == "<fill>":
        return False
    return expected.lower() in answer.lower()


def sources_match(citations: List[str], expected_sources: List[str]) -> float:
    if not expected_sources:
        # If no expected sources (negative case), treat as correct if citations empty or unrelated
        return 1.0 if not citations else 0.0
    if not citations:
        return 0.0
    hits = 0
    for es in expected_sources:
        es_low = es.lower()
        if any(es_low in c.lower() for c in citations):
            hits += 1
    return hits / max(1, len(expected_sources))


def avg_top_score(retrieved: List[Tuple[Dict[str, Any], float]], top_n: int = 3) -> float:
    if not retrieved:
        return 0.0
    scores = [s for (_m, s) in retrieved[:top_n]]
    return sum(scores) / len(scores)


def run_eval(docs_dir: str, dataset_path: str, k: int, n: int, threshold: float) -> Dict[str, Any]:
    # Ingest and index
    t0 = time.time()
    ing = ingest_dir(docs_dir)
    idx = VectorIndex()
    idx.build(ing["chunks"]) if ing.get("chunks") else None
    build_s = time.time() - t0

    data = load_dataset(dataset_path)
    results = []

    total_latency = 0.0
    accuracy_hits = 0
    source_acc_hits = 0.0
    fallback_count = 0

    for item in data:
        q = item.get("question", "")
        expected_ans = item.get("expected_answer", "")
        expected_srcs = item.get("expected_sources", [])

        t_q = time.time()
        retrieved = retrieve_chunks(q, index=idx, top_k=k, rerank_top_n=n, similarity_threshold=threshold) if idx else []
        answer, citations = generate_answer(q, retrieved)
        latency = time.time() - t_q

        total_latency += latency
        acc = contains_expected(answer, expected_ans)
        if acc:
            accuracy_hits += 1
        src_match = sources_match(citations, expected_srcs)
        source_acc_hits += src_match

        # crude fallback detection
        lower = answer.lower()
        is_fallback = (
            ("couldn't find" in lower or "limited matches" in lower) and len(retrieved) == 0
        )
        if is_fallback:
            fallback_count += 1

        results.append({
            "question": q,
            "expected_answer": expected_ans,
            "answer": answer,
            "citations": citations,
            "retrieved_scores": [float(s) for (_m, s) in retrieved],
            "avg_top3_score": avg_top_score(retrieved),
            "latency_s": latency,
            "answer_correct": acc,
            "source_match": src_match,
        })

    n_items = max(1, len(data))
    report = {
        "docs_indexed": ing.get("num_docs", 0),
        "chunks_indexed": ing.get("num_chunks", 0),
        "build_time_s": build_s,
        "questions": len(data),
        "accuracy": accuracy_hits / n_items,
        "source_correctness": source_acc_hits / n_items,
        "avg_latency_s": total_latency / n_items,
        "fallback_rate": fallback_count / n_items,
        "k": k,
        "rerank_top_n": n,
        "similarity_threshold": threshold,
        "results": results,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG chatbot on a dataset")
    parser.add_argument("--docs", required=True, help="Path to directory containing PDF/TXT docs")
    parser.add_argument("--dataset", required=True, help="Path to evaluation dataset .jsonl")
    parser.add_argument("--k", type=int, default=8, help="Initial top-k for vector search")
    parser.add_argument("--n", type=int, default=5, help="Top-N after reranking")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    parser.add_argument("--out", type=str, default="evaluation_report.json", help="Output JSON report path")

    args = parser.parse_args()
    report = run_eval(args.docs, args.dataset, args.k, args.n, args.threshold)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to {out_path.resolve()}")
    print("Summary:")
    print(json.dumps({k: report[k] for k in [
        "docs_indexed", "chunks_indexed", "questions", "accuracy", "source_correctness", "avg_latency_s", "fallback_rate"
    ]}, indent=2))


if __name__ == "__main__":
    main()
