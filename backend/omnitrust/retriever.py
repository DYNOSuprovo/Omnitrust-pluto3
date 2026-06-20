"""
OmniTrust-RAG Document Retriever.

Provides three retriever classes:

* ``WikipediaRetriever`` – fetches pages via the ``wikipedia`` library.
* ``CorpusRetriever``    – searches an in-memory document store with TF-IDF.
* ``HybridRetriever``    – combines both and optionally re-ranks via NVIDIA NIM.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from collections import Counter
from typing import Any

import httpx
import wikipedia

from omnitrust.config import (
    NVIDIA_API_KEY,
    NVIDIA_RANKING_URL,
    NVIDIA_RERANK_MODEL,
    RETRIEVER_TOP_K,
)

logger = logging.getLogger(__name__)

# Wikipedia library configuration
wikipedia.set_lang("en")
wikipedia.set_rate_limiting(True)
wikipedia.set_user_agent("OmniTrust-RAG/1.0 (research project; contact@example.com)")


# ---------------------------------------------------------------------------
# Helpers – lightweight TF-IDF with no external dependencies
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will "
    "would shall should may might can could of in to for on with at by from "
    "as into through during before after above below between out off over "
    "up about than so that this these those it its he she they them we you "
    "i me my our your his her their what which who whom how when where why "
    "not no nor and or but if".split()
)


def _tokenize(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


def _term_freq(tokens: list[str]) -> dict[str, float]:
    counter = Counter(tokens)
    total = len(tokens) or 1
    return {t: c / total for t, c in counter.items()}


def _idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    n = len(corpus_tokens)
    df: dict[str, int] = {}
    for tokens in corpus_tokens:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}


def _tfidf_vector(tf: dict[str, float], idf: dict[str, float]) -> dict[str, float]:
    return {t: tf_val * idf.get(t, 1.0) for t, tf_val in tf.items()}


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _doc_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Wikipedia Retriever
# ---------------------------------------------------------------------------


class WikipediaRetriever:
    """Fetches Wikipedia pages, caching results to avoid redundant network calls."""

    def __init__(self) -> None:
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self.api_url = "https://en.wikipedia.org/w/api.php"
        self.headers = {
            "User-Agent": "OmniTrust-RAG/1.0 (research project; contact@example.com)"
        }

    def _get_with_retry(
        self, client: httpx.Client, url: str, params: dict[str, Any]
    ) -> httpx.Response:
        """Fetch using client GET with exponential backoff on 429 rate limits."""
        import time
        delay = 1.5
        max_retries = 3
        time.sleep(0.25)  # Pace requests
        for attempt in range(max_retries):
            try:
                r = client.get(url, params=params, headers=self.headers)
                if r.status_code == 429:
                    logger.warning(
                        "Wikipedia API rate limited (429) on attempt %d/%d. Waiting %.1fs...",
                        attempt + 1, max_retries, delay
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                r.raise_for_status()
                return r
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    logger.warning(
                        "Wikipedia API rate limited (429) on attempt %d/%d. Waiting %.1fs...",
                        attempt + 1, max_retries, delay
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                raise
            except httpx.RequestError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2.0
        return client.get(url, params=params, headers=self.headers)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        import urllib.parse
        cache_key = query.strip().lower()
        if cache_key in self._cache:
            logger.debug("WikipediaRetriever cache hit for '%s'", query)
            return self._cache[cache_key][: top_k]

        results: list[dict[str, Any]] = []
        try:
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": top_k
            }
            with httpx.Client(timeout=10.0) as client:
                r = self._get_with_retry(client, self.api_url, search_params)
                r.raise_for_status()
                data = r.json()
                search_results = data.get("query", {}).get("search", [])
                titles = [item["title"] for item in search_results]

                for title in titles:
                    fetch_params = {
                        "action": "query",
                        "prop": "extracts",
                        "exintro": 1,
                        "explaintext": 1,
                        "titles": title,
                        "format": "json",
                        "redirects": 1
                    }
                    rf = self._get_with_retry(client, self.api_url, fetch_params)
                    rf.raise_for_status()
                    fdata = rf.json()
                    pages = fdata.get("query", {}).get("pages", {})
                    for page_id, page in pages.items():
                        if page_id == "-1":
                            continue
                        extract = page.get("extract", "")
                        if not extract:
                            continue
                        safe_title = urllib.parse.quote(title.replace(" ", "_"))
                        page_url = f"https://en.wikipedia.org/wiki/{safe_title}"
                        doc = {
                            "id": _doc_id(page_url),
                            "title": page.get("title", title),
                            "text": extract[:2500],
                            "source": page_url
                        }
                        results.append(doc)
        except Exception:
            logger.warning("Wikipedia search failed for query: %s", query, exc_info=True)
            self._cache[cache_key] = []
            return []

        self._cache[cache_key] = results
        return results[: top_k]


# ---------------------------------------------------------------------------
# Corpus Retriever (in-memory TF-IDF)
# ---------------------------------------------------------------------------


class CorpusRetriever:
    """Stores an in-memory document corpus and searches it via TF-IDF."""

    def __init__(self) -> None:
        self._documents: list[dict[str, Any]] = []
        self._tokens: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._tfidf_vecs: list[dict[str, float]] = []

    @property
    def documents(self) -> list[dict[str, Any]]:
        return list(self._documents)

    def add_documents(self, docs: list[dict[str, Any]]) -> None:
        """Add documents and rebuild the TF-IDF index."""
        for doc in docs:
            doc_record = {
                "id": doc.get("id", _doc_id(doc.get("text", ""))),
                "title": doc.get("title", "Untitled"),
                "text": doc.get("text", ""),
                "source": doc.get("source", "corpus"),
            }
            self._documents.append(doc_record)
            self._tokens.append(_tokenize(doc_record["text"]))

        # Rebuild IDF and TF-IDF vectors for the entire corpus
        self._idf = _idf(self._tokens)
        self._tfidf_vecs = [
            _tfidf_vector(_term_freq(toks), self._idf) for toks in self._tokens
        ]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._documents:
            return []

        q_tokens = _tokenize(query)
        q_tf = _term_freq(q_tokens)
        q_vec = _tfidf_vector(q_tf, self._idf)

        scored = [
            (idx, _cosine_sim(q_vec, dv))
            for idx, dv in enumerate(self._tfidf_vecs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[dict[str, Any]] = []
        for idx, score in scored[:top_k]:
            if score > 0.0:
                doc = dict(self._documents[idx])
                doc["_tfidf_score"] = round(score, 4)
                results.append(doc)
        return results


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Combines Wikipedia and corpus search, with optional NVIDIA NIM reranking."""

    def __init__(self, use_reranker: bool = True) -> None:
        self.wiki = WikipediaRetriever()
        self.corpus = CorpusRetriever()
        self.use_reranker = use_reranker

    def search(self, query: str, top_k: int = RETRIEVER_TOP_K) -> list[dict[str, Any]]:
        # Run both retrievers
        wiki_docs = self.wiki.search(query, top_k=top_k)
        corpus_docs = self.corpus.search(query, top_k=top_k)

        # Merge and deduplicate by id
        seen_ids: set[str] = set()
        merged: list[dict[str, Any]] = []
        for doc in wiki_docs + corpus_docs:
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                merged.append(doc)

        if not merged:
            return []

        # Optionally rerank with NVIDIA NIM
        if self.use_reranker and len(merged) > 1:
            reranked = self._nvidia_rerank(query, merged, top_k)
            if reranked is not None:
                return reranked

        return merged[:top_k]

    def _nvidia_rerank(
        self, query: str, documents: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]] | None:
        """Call the NVIDIA NIM reranker endpoint.  Returns ``None`` on failure."""
        passages = [
            {"text": doc.get("text", "")[:1500]}
            for doc in documents
        ]
        payload = {
            "model": NVIDIA_RERANK_MODEL,
            "query": {"text": query},
            "passages": passages,
            "top_n": min(top_k, len(documents)),
        }
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            t0 = time.perf_counter()
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(NVIDIA_RANKING_URL, json=payload, headers=headers)
            elapsed = time.perf_counter() - t0
            logger.info("NVIDIA rerank responded in %.2fs (status %d)", elapsed, resp.status_code)

            if resp.status_code != 200:
                logger.warning("NVIDIA rerank non-200: %s %s", resp.status_code, resp.text[:500])
                return None

            data = resp.json()
            rankings = data.get("rankings", [])
            if not rankings:
                return None

            # Sort by logit descending and map back to original documents
            rankings.sort(key=lambda r: r.get("logit", 0.0), reverse=True)
            reranked: list[dict[str, Any]] = []
            for entry in rankings:
                idx = entry.get("index", 0)
                if 0 <= idx < len(documents):
                    doc = dict(documents[idx])
                    doc["_rerank_logit"] = round(entry.get("logit", 0.0), 4)
                    reranked.append(doc)
            return reranked[:top_k]

        except Exception:  # noqa: BLE001
            logger.warning("NVIDIA rerank call failed", exc_info=True)
            return None
