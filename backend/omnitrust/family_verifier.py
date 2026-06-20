"""
OmniTrust-RAG Family-Attention Verification Engine.

Implements a multi-head "Family Attention" mechanism that evaluates claims
from four complementary perspectives:

1. **Semantic Head**   – lexical overlap and synonym-aware matching.
2. **Temporal/Metric** – detection and comparison of dates, numbers, units.
3. **Named Entity**    – proper-noun and entity overlap.
4. **Context Head**    – broader contextual coherence via bag-of-words.

Each head produces a distribution over {supported, uncertain, unsupported}.
The engine then computes:
* A gated mixing of heads into a single *consistency score*.
* A lateral *similarity matrix* showing inter-head agreement.
* The *Jensen–Shannon Divergence* to quantify head diversity.

All maths is pure Python (``math`` module only).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# Text helpers
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

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")
_DATE_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


def _tokenize(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


def _bag(tokens: list[str]) -> set[str]:
    return set(tokens)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _overlap_coeff(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _extract_numbers(text: str) -> list[float]:
    matches = _NUMBER_RE.findall(text)
    nums: list[float] = []
    for m in matches:
        try:
            nums.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return nums


def _extract_dates(text: str) -> list[str]:
    return [m.group(0).lower() for m in _DATE_RE.finditer(text)]


def _extract_entities(text: str) -> set[str]:
    return {m.group(0) for m in _ENTITY_RE.finditer(text)}


# ---------------------------------------------------------------------------
# Attention Heads
# ---------------------------------------------------------------------------

def _semantic_head(claim_tokens: set[str], evidence_tokens: set[str]) -> tuple[float, float, float]:
    """Semantic overlap head.  Returns (supported, uncertain, unsupported)."""
    sim = _jaccard(claim_tokens, evidence_tokens)
    if sim >= 0.5:
        return (sim, 1.0 - sim, 0.0)
    elif sim >= 0.2:
        return (sim, 0.6, 0.4 - sim)
    else:
        return (sim, 0.3, 0.7 - sim)


def _temporal_metric_head(claim_text: str, evidence_text: str) -> tuple[float, float, float]:
    """Date and number comparison head."""
    claim_nums = _extract_numbers(claim_text)
    evidence_nums = _extract_numbers(evidence_text)
    claim_dates = _extract_dates(claim_text)
    evidence_dates = _extract_dates(evidence_text)

    # Number agreement
    num_score = 0.5  # default uncertain
    if claim_nums and evidence_nums:
        matches = sum(1 for cn in claim_nums if any(abs(cn - en) / max(abs(en), 1e-9) < 0.05 for en in evidence_nums))
        num_score = matches / len(claim_nums) if claim_nums else 0.5

    # Date agreement
    date_score = 0.5
    if claim_dates and evidence_dates:
        claim_date_set = set(claim_dates)
        evidence_date_set = set(evidence_dates)
        overlap = len(claim_date_set & evidence_date_set)
        date_score = overlap / len(claim_date_set) if claim_date_set else 0.5

    # Combine
    if claim_nums or claim_dates:
        combined = 0.0
        weight = 0.0
        if claim_nums:
            combined += num_score
            weight += 1.0
        if claim_dates:
            combined += date_score
            weight += 1.0
        combined /= weight
        supported = combined
        unsupported = max(0.0, 1.0 - combined - 0.2)
        uncertain = 1.0 - supported - unsupported
    else:
        # No temporal/metric content – maximum uncertainty
        supported = 0.3
        uncertain = 0.5
        unsupported = 0.2

    # Normalise to sum to 1
    total = supported + uncertain + unsupported
    if total > 0:
        supported /= total
        uncertain /= total
        unsupported /= total

    return (supported, uncertain, unsupported)


def _entity_head(claim_text: str, evidence_text: str) -> tuple[float, float, float]:
    """Named-entity overlap head."""
    claim_ents = _extract_entities(claim_text)
    evidence_ents = _extract_entities(evidence_text)

    if not claim_ents:
        return (0.3, 0.5, 0.2)

    overlap = _overlap_coeff(claim_ents, evidence_ents)
    if overlap >= 0.6:
        return (overlap, 1.0 - overlap, 0.0)
    elif overlap >= 0.2:
        return (overlap, 0.5, 0.5 - overlap)
    else:
        return (overlap, 0.3, 0.7 - overlap)


def _context_head(claim_tokens: set[str], evidence_tokens: set[str]) -> tuple[float, float, float]:
    """Broader bag-of-words contextual coherence head."""
    sim = _jaccard(claim_tokens, evidence_tokens)
    # Softer thresholds than semantic head
    supported = sim * 0.8
    uncertain = 0.4 * (1.0 - sim)
    unsupported = max(0.0, 1.0 - supported - uncertain)
    total = supported + uncertain + unsupported
    if total > 0:
        supported /= total
        uncertain /= total
        unsupported /= total
    return (supported, uncertain, unsupported)


# ---------------------------------------------------------------------------
# Jensen–Shannon Divergence
# ---------------------------------------------------------------------------

def _kl_div(p: list[float], q: list[float]) -> float:
    """KL(P || Q) with epsilon smoothing."""
    eps = 1e-12
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q))


def _js_divergence(distributions: list[list[float]]) -> float:
    """Generalised Jensen–Shannon Divergence over *n* distributions."""
    n = len(distributions)
    if n <= 1:
        return 0.0
    dim = len(distributions[0])
    # Mean distribution
    m = [sum(distributions[i][j] for i in range(n)) / n for j in range(dim)]
    jsd = sum(_kl_div(distributions[i], m) for i in range(n)) / n
    return jsd


# ---------------------------------------------------------------------------
# Similarity matrix between head identity vectors
# ---------------------------------------------------------------------------

def _dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(ai * ai for ai in a))


def _cosine_list(a: list[float], b: list[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


# ---------------------------------------------------------------------------
# Family Attention Verifier
# ---------------------------------------------------------------------------

_HEAD_NAMES = ["Semantic", "Temporal/Metric", "Named Entity", "Context"]
_HEAD_WEIGHTS = [0.35, 0.20, 0.20, 0.25]  # gating weights


class FamilyAttentionVerifier:
    """Multi-head verification engine.

    Usage::

        verifier = FamilyAttentionVerifier()
        result = verifier.verify(claims, evidence)
    """

    def verify(
        self,
        claims: list[str],
        evidence: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Run all heads over every (claim, evidence) pair and aggregate.

        Parameters
        ----------
        claims:
            List of claim strings extracted from the answer.
        evidence:
            List of evidence dicts, each with at least a ``"text"`` key.

        Returns
        -------
        dict with ``consistency_score``, ``js_divergence``,
        ``head_perspectives``, ``similarity_matrix``.
        """
        if not claims or not evidence:
            return {
                "consistency_score": 0.0,
                "js_divergence": 0.0,
                "head_perspectives": [
                    {"head_id": i, "name": _HEAD_NAMES[i], "supported": 0.0, "uncertain": 1.0, "unsupported": 0.0}
                    for i in range(4)
                ],
                "similarity_matrix": [[1.0] * 4 for _ in range(4)],
            }

        # Concatenate all evidence text
        full_evidence = " ".join(e.get("text", "") for e in evidence)
        evidence_tokens = _bag(_tokenize(full_evidence))

        # Accumulate head outputs across all claims
        head_accumulators: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(4)]
        num_claims = len(claims)

        for claim in claims:
            claim_tokens = _bag(_tokenize(claim))

            sem = _semantic_head(claim_tokens, evidence_tokens)
            tmp = _temporal_metric_head(claim, full_evidence)
            ent = _entity_head(claim, full_evidence)
            ctx = _context_head(claim_tokens, evidence_tokens)

            for head_idx, scores in enumerate([sem, tmp, ent, ctx]):
                for k in range(3):
                    head_accumulators[head_idx][k] += scores[k]

        # Average across claims
        head_distributions: list[list[float]] = []
        for acc in head_accumulators:
            total = sum(acc)
            if total > 0:
                dist = [v / total for v in acc]
            else:
                dist = [0.0, 1.0, 0.0]
            head_distributions.append(dist)

        # Build head perspectives
        head_perspectives: list[dict[str, Any]] = []
        for i in range(4):
            head_perspectives.append(
                {
                    "head_id": i,
                    "name": _HEAD_NAMES[i],
                    "supported": round(head_distributions[i][0], 4),
                    "uncertain": round(head_distributions[i][1], 4),
                    "unsupported": round(head_distributions[i][2], 4),
                }
            )

        # Gated mixing → consistency score
        consistency = sum(
            _HEAD_WEIGHTS[i] * head_distributions[i][0] for i in range(4)
        )
        consistency = round(max(0.0, min(1.0, consistency)), 4)

        # Jensen–Shannon Divergence
        jsd = round(_js_divergence(head_distributions), 6)

        # Lateral similarity matrix (4×4)
        sim_matrix: list[list[float]] = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                sim_matrix[i][j] = round(_cosine_list(head_distributions[i], head_distributions[j]), 4)

        return {
            "consistency_score": consistency,
            "js_divergence": jsd,
            "head_perspectives": head_perspectives,
            "similarity_matrix": sim_matrix,
        }
