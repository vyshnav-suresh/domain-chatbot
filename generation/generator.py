from __future__ import annotations

import os
import time
import logging
from typing import List, Dict, Any, Tuple, Iterator

try:
    import google.generativeai as genai
except Exception:
    genai = None  # type: ignore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None  # type: ignore


# Module logger
log = logging.getLogger("generation")


def _format_citations(sources: List[Tuple[Dict[str, Any], float]]) -> List[str]:
    cites: List[str] = []
    for idx, (meta, score) in enumerate(sources, start=1):
        doc = meta.get("document", "UnknownDoc")
        ch = meta.get("chunk_index", meta.get("chunk_id", "?"))
        page = meta.get("page")
        section_id = meta.get("section_id")
        hint = (meta.get("section_hint") or "").strip()
        if hint:
            # keep it short
            hint = " ".join(hint.split())[:120]
        # Build display string with optional page/section pair in user's style and context index
        loc = None
        if page is not None and section_id:
            loc = f"[page {page},{section_id}]"
        elif page is not None:
            loc = f"[page {page}]"
        elif section_id:
            loc = f"[sec {section_id}]"
        header = f"{doc} {loc}" if loc else f"{doc}"
        ctx_tag = f" [Context {idx}]"
        tail_hint = f" — {hint}" if hint else ""
        cites.append(f"{header}{tail_hint}{ctx_tag} (score={score:.2f})")
    return cites


def _build_prompt(query: str, contexts: List[str]) -> str:
    ctx = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts))
    prompt = (
    "You are a domain-specific assistant. Always answer the user's question "
    "using ONLY the provided context below. Do not use outside knowledge.\n\n"
    "Style guidelines:\n"
    "- Write in a friendly, conversational tone (first person, natural phrasing).\n"
    "- Use short paragraphs and bullet points when helpful.\n"
    "- If the context does not provide enough information, be transparent and ask "
    "the user for clarification instead of guessing.\n"
    "- Support all claims with inline citations using the numbered context blocks below.\n"
    "  Cite as [Context N] (e.g., [Context 1], [Context 2]) right after the relevant statement.\n"
    "- Be concise and precise, avoid unnecessary detail.\n\n"
    f"Context:\n{ctx}\n\n"
    f"Question: {query}\n\n"
    "Answer (include citations):"
)
    return prompt


def _fallback_answer(query: str, sources: List[Tuple[Dict[str, Any], float]]) -> str:
    def _clarify_list() -> str:
        hints = [
            "Which document or section should I focus on?",
            "Are you referring to a specific version, dataset, or timeframe?",
            "Do you want a definition, a step-by-step process, or a comparison?",
        ]
        bullets = "\n".join(f"- {h}" for h in hints)
        return bullets

    if not sources:
        # No strong matches — respond gently and invite clarification
        return (
            f"I’m not seeing a direct match for “{query}” in the indexed passages. "
            "Here are a couple of ways we can narrow it down:\n\n"
            f"{_clarify_list()}"
        )
    # Use top context as partial info
    top_texts = [meta["text"] for meta, _ in sources[:2]]
    excerpt = "\n".join(t[:600] for t in top_texts if t)
    return (
        "Here’s the closest information I could find so far, which may be related:\n\n"
        f"{excerpt}\n\n"
        "If you can share a bit more detail (document/section, or the exact aspect you need), I’ll refine the answer."
    )


def _format_markdown_answer(answer: str, confidence: float | None = None) -> str:
    """Wrap the model/fallback answer into a friendly Markdown template."""
    parts = [answer.strip()]
    if confidence is not None:
        conf_str = f"{confidence:.1f}/10" if isinstance(confidence, (int, float)) else str(confidence)
        parts.append(f"\n*Confidence: {conf_str}*")
    # Add a light nudge for clarification when confidence is low
    if isinstance(confidence, (int, float)) and confidence < 7.0:
        parts.append("\n*If you can share a tad more detail (scope, section/page, exact term), I can tighten this up further.*")
    return "\n\n".join(parts)


def generate_answer(query: str, sources: List[Tuple[Dict[str, Any], float]]):
    """
    Generate a concise, cited answer using Google Gemini if available; otherwise use a safe fallback.
    Env vars: GOOGLE_API_KEY (preferred) or GEMINI_API_KEY.
    Returns (answer_text, citation_list)
    """
    citations = _format_citations(sources)
    contexts = [meta["text"] for meta, _ in sources[:5]]

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    use_gemini = genai is not None and api_key and len(contexts) > 0
    log.info("[GEN] generate_answer use_gemini=%s contexts=%d", bool(use_gemini), len(contexts))

    if use_gemini:
        try:
            genai.configure(api_key=api_key)
            # Choose a fast, cost-effective model; switch to pro if needed.
            t0 = time.perf_counter()
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = _build_prompt(query, contexts)
            resp = model.generate_content(prompt)
            log.info("[GEN] Gemini generate_content in %.2fs", time.perf_counter() - t0)
            raw_answer = (resp.text or "").strip()

            # Self-assessment for uncertainty
            conf_prompt = (
                "On a scale of 1-10, how confident are you in your previous answer? "
                "Return only a single number."
            )
            t1 = time.perf_counter()
            conf_resp = model.generate_content(conf_prompt)
            log.debug("[GEN] Confidence query completed in %.2fs", time.perf_counter() - t1)
            try:
                conf_str = (conf_resp.text or "").strip()
                conf = float("".join(c for c in conf_str if (c.isdigit() or c == '.')) or 7.0)
            except Exception:
                conf = 7.0
            if not raw_answer:
                # If for any reason empty, fallback
                raise ValueError("Empty answer from Gemini")
            # If confidence is low, append gentle clarification prompts
            if conf < 7.0:
                raw_answer += ("\n\n" "If it helps, tell me the document/section or whether you want a definition, a how-to, or a comparison.")
            formatted = _format_markdown_answer(raw_answer, confidence=conf)
            log.info("[GEN] Model answer produced (len=%d, conf=%.2f)", len(formatted), conf)
            return formatted, citations
        except Exception as e:
            # Fall back to extractive mode
            log.exception("[GEN] Gemini path failed, falling back: %s", e)

    # Fallback path
    fallback = _fallback_answer(query, sources)
    formatted = _format_markdown_answer(fallback, confidence=None)
    log.info("[GEN] Fallback answer produced (len=%d)", len(formatted))
    return formatted, citations


def stream_answer(query: str, sources: List[Tuple[Dict[str, Any], float]]) -> Tuple[Iterator[str], List[str]]:
    """
    Stream an answer using Gemini if available. Returns (iterator of text chunks, citations list).
    Fallback: yields a single fallback answer chunk.
    """
    citations = _format_citations(sources)
    contexts = [meta["text"] for meta, _ in sources[:5]]

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    use_gemini = genai is not None and api_key and len(contexts) > 0

    def _fallback_iter() -> Iterator[str]:
        yield _fallback_answer(query, sources)

    if use_gemini:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = _build_prompt(query, contexts)
            t0 = time.perf_counter()
            resp = model.generate_content(prompt, stream=True)
            log.info("[GEN] Gemini streaming started in %.2fs", time.perf_counter() - t0)

            def _iter() -> Iterator[str]:
                try:
                    for chunk in resp:
                        # Each chunk may contain .text
                        if hasattr(chunk, "text") and chunk.text:
                            yield chunk.text
                finally:
                    # Ensure stream is closed if needed
                    try:
                        resp.close()  # type: ignore[attr-defined]
                    except Exception:
                        pass
            return _iter(), citations
        except Exception as e:
            # If streaming fails, fall back to non-stream
            log.exception("[GEN] Streaming failed, falling back: %s", e)
            return _fallback_iter(), citations


def suggest_questions(chunks: List[Dict[str, Any]], min_q: int = 5, max_q: int = 10) -> List[str]:
    """
    Generate suggested questions based on the provided chunks.
    Tries Gemini first; falls back to TF-IDF keyword-based templates.
    """
    min_q = max(5, min_q)
    max_q = min(10, max_q)
    max_q = max(min_q, max_q)
    texts = [c.get("text", "") for c in chunks if c.get("text")]
    if not texts:
        return []

    # Limit context to avoid long prompts
    context_sample = "\n\n".join(texts[:10])[:6000]

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if genai is not None and api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "You are given excerpts from a document corpus. "
                "Propose between {min_q} and {max_q} concise, useful user questions that someone might ask to retrieve information from these documents. "
                "Questions should be domain-relevant, varied (definitions, procedures, comparisons, limitations, requirements), and not trivial. "
                "Return them as a JSON array of strings only, no extra text.\n\n"
                f"Excerpts:\n{context_sample}\n\n"
                f"Number of questions: between {min_q} and {max_q}."
            )
            t0 = time.perf_counter()
            resp = model.generate_content(prompt)
            log.info("[SUGGEST] Gemini generated suggestions in %.2fs", time.perf_counter() - t0)
            raw = (resp.text or "").strip()
            # Try to parse as JSON array
            import json
            try:
                arr = json.loads(raw)
                if isinstance(arr, list):
                    arr = [str(x).strip() for x in arr if str(x).strip()]
                    return arr[:max_q]
            except Exception:
                log.debug("[SUGGEST] JSON parse failed; falling back to line split")
            # Fallback: split by lines / bullets
            lines = [ln.strip("- •\t ") for ln in raw.splitlines() if ln.strip()]
            lines = [ln for ln in lines if len(ln) > 8]
            return lines[:max_q] if lines else []
        except Exception as e:
            log.exception("[SUGGEST] Gemini path failed; trying TF-IDF fallback: %s", e)

    # TF-IDF fallback
    if TfidfVectorizer is None:
        return []
    try:
        vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
        t0 = time.perf_counter()
        X = vectorizer.fit_transform(texts)
        log.info("[SUGGEST] TF-IDF fit_transform on %d texts in %.2fs", len(texts), time.perf_counter() - t0)
        scores = X.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        pairs = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)[:30]
        keywords = [t for t, _ in pairs]
        # Template questions
        templates = [
            "What is {k}?",
            "How does {k} work?",
            "What are the key components of {k}?",
            "What are the requirements for {k}?",
            "What are the limitations of {k}?",
            "How to configure {k}?",
            "What is the difference between {k1} and {k2}?",
        ]
        qs: List[str] = []
        # Single-keyword questions
        for k in keywords[:6]:
            for t in templates[:6]:
                q = t.format(k=k)
                if q not in qs:
                    qs.append(q)
                    if len(qs) >= max_q - 2:
                        break
            if len(qs) >= max_q - 2:
                break
        # Comparison questions
        if len(keywords) >= 2 and len(qs) < max_q:
            qs.append(templates[6].format(k1=keywords[0], k2=keywords[1]))
        if len(keywords) >= 4 and len(qs) < max_q:
            qs.append(templates[6].format(k1=keywords[2], k2=keywords[3]))
        log.info("[SUGGEST] TF-IDF produced %d suggestion(s)", len(qs[:max_q] if len(qs) >= min_q else qs))
        return qs[:max_q] if len(qs) >= min_q else qs
    except Exception as e:
        log.exception("[SUGGEST] TF-IDF fallback failed: %s", e)
        return []
