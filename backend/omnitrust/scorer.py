"""
OmniTrust-RAG Evidence Quality Scorers.

Two scoring components quantify the diversity and usefulness of retrieved
evidence **before** it enters the verification and synthesis stages:

* ``IndependenceScorer`` – clusters near-duplicate documents so that the
  pipeline can down-weight redundant sources.
* ``UtilityScorer`` – measures per-document novelty, contradiction risk,
  and overall utility relative to the original question.

All maths uses only the Python ``math`` module (no NumPy / sklearn needed).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from omnitrust.config import INDEPENDENCE_SIMILARITY_THRESHOLD

# ---------------------------------------------------------------------------
# Text helpers (shared with retriever but duplicated here to keep the
# module self-contained).
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

_NEGATION_WORDS = frozenset(
    "not never no none neither nor cannot couldn't didn't doesn't don't "
    "hasn't haven't hadn't isn't aren't wasn't weren't won't wouldn't "
    "shouldn't mustn't false incorrect wrong untrue inaccurate deny denied "
    "denies refute refuted refutes contradict contradicted contradicts".split()
)


def _tokenize(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


def _term_freq(tokens: list[str]) -> dict[str, float]:
    counter = Counter(tokens)
    total = len(tokens) or 1
    return {t: c / total for t, c in counter.items()}


def _idf_from_docs(corpus_tokens: list[list[str]]) -> dict[str, float]:
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


# ---------------------------------------------------------------------------
# Independence Scorer
# ---------------------------------------------------------------------------


class IndependenceScorer:
    """Clusters near-duplicate documents and assigns independence scores.

    Independence = 1 / cluster_size  (unique doc → 1.0, duplicate → < 1.0).
    Similarity threshold defaults to ``INDEPENDENCE_SIMILARITY_THRESHOLD``
    from config (0.85).
    """

    def __init__(self, threshold: float = INDEPENDENCE_SIMILARITY_THRESHOLD) -> None:
        self.threshold = threshold

    def score(self, documents: list[str]) -> list[dict[str, Any]]:
        """Score a list of document texts.

        Returns a list of dicts parallel to *documents*, each containing:
        ``doc_id``, ``independence_score``, ``cluster_id``, ``cluster_size``,
        ``is_duplicate``.
        """
        n = len(documents)
        if n == 0:
            return []

        # Tokenize & build TF-IDF
        all_tokens = [_tokenize(d) for d in documents]
        idf = _idf_from_docs(all_tokens)
        vectors = [_tfidf_vector(_term_freq(t), idf) for t in all_tokens]

        # Pairwise similarity matrix
        sim_matrix: list[list[float]] = [
            [0.0] * n for _ in range(n)
        ]
        for i in range(n):
            sim_matrix[i][i] = 1.0
            for j in range(i + 1, n):
                s = _cosine_sim(vectors[i], vectors[j])
                sim_matrix[i][j] = s
                sim_matrix[j][i] = s

        # Greedy single-linkage clustering
        cluster_ids = list(range(n))  # each doc starts in its own cluster
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] >= self.threshold:
                    # Merge clusters
                    old_cid = cluster_ids[j]
                    new_cid = cluster_ids[i]
                    for k in range(n):
                        if cluster_ids[k] == old_cid:
                            cluster_ids[k] = new_cid

        # Compute cluster sizes
        cluster_sizes: dict[int, int] = Counter(cluster_ids)

        results: list[dict[str, Any]] = []
        for idx in range(n):
            cid = cluster_ids[idx]
            csize = cluster_sizes[cid]
            results.append(
                {
                    "doc_id": idx,
                    "independence_score": round(1.0 / csize, 4),
                    "cluster_id": cid,
                    "cluster_size": csize,
                    "is_duplicate": csize > 1,
                }
            )
        return results


# ---------------------------------------------------------------------------
# Utility Scorer
# ---------------------------------------------------------------------------


class UtilityScorer:
    """Measures per-document novelty, contradiction risk, and composite utility.

    Utility = 0.5 × novelty + 0.3 × (1 − contradiction) + 0.2 × length_bonus
    """

    def score(
        self, question: str, documents: list[str]
    ) -> list[dict[str, Any]]:
        """Score a list of document texts relative to the *question*.

        Returns a list of dicts parallel to *documents*, each containing:
        ``doc_id``, ``novelty``, ``contradiction``, ``utility_score``,
        ``is_useful``.
        """
        n = len(documents)
        if n == 0:
            return []

        all_tokens = [_tokenize(d) for d in documents]
        idf = _idf_from_docs(all_tokens)
        vectors = [_tfidf_vector(_term_freq(t), idf) for t in all_tokens]

        # Question vector (for relevance-boosted novelty)
        q_tokens = _tokenize(question)
        q_vec = _tfidf_vector(_term_freq(q_tokens), idf)

        results: list[dict[str, Any]] = []
        for idx in range(n):
            # --- Novelty ---
            # Average cosine *distance* from all prior documents
            if idx == 0:
                novelty = 1.0  # first doc is inherently novel
            else:
                sims = [_cosine_sim(vectors[idx], vectors[j]) for j in range(idx)]
                avg_sim = sum(sims) / len(sims)
                novelty = 1.0 - avg_sim

            # Boost novelty if document is relevant to the question
            q_relevance = _cosine_sim(vectors[idx], q_vec)
            novelty = min(1.0, novelty * 0.7 + q_relevance * 0.3)

            # --- Contradiction ---
            doc_lower = documents[idx].lower()
            doc_word_count = len(doc_lower.split()) or 1
            neg_count = sum(1 for w in _NEGATION_WORDS if w in doc_lower)
            contradiction = min(1.0, neg_count / (doc_word_count * 0.05 + 1))

            # --- Length bonus ---
            # Longer documents (up to a point) are more informative
            length = len(documents[idx])
            length_bonus = min(1.0, length / 2000.0)

            # --- Composite utility ---
            utility = 0.5 * novelty + 0.3 * (1.0 - contradiction) + 0.2 * length_bonus
            utility = round(max(0.0, min(1.0, utility)), 4)

            results.append(
                {
                    "doc_id": idx,
                    "novelty": round(novelty, 4),
                    "contradiction": round(contradiction, 4),
                    "utility_score": utility,
                    "is_useful": utility >= 0.3,
                }
            )
        return results
