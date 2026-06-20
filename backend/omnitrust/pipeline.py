"""
OmniTrust-RAG Orchestration Pipeline.

Coordinates every component in a deterministic 10-step sequence and returns
a fully populated ``PipelineResponse``.

Steps:
 1. Plan search queries              (Planner agent)
 2. Strategist audit                  (Strategist agent)
 3. Retrieve documents                (Hybrid retriever)
 4. Score independence                (IndependenceScorer)
 5. Score utility                     (UtilityScorer)
 6. Filter documents                  (remove duplicates / low utility)
 7. Family-attention verification     (FamilyAttentionVerifier)
 8. Critic checks claims              (Critic agent)
 9. Synthesise final answer           (Synthesizer agent)
10. Assemble response                 (this module)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from omnitrust.agents import CriticAgent, PlannerAgent, StrategistAgent, SynthesizerAgent
from omnitrust.bus import MessageBus
from omnitrust.family_verifier import FamilyAttentionVerifier
from omnitrust.models import (
    AgentLog,
    CheckedClaim,
    DocumentResult,
    HeadPerspective,
    PipelineResponse,
    VerificationResult,
)
from omnitrust.retriever import HybridRetriever
from omnitrust.scorer import IndependenceScorer, UtilityScorer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple claim extraction (sentence-level)
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _extract_claims(text: str, max_claims: int = 8) -> list[str]:
    """Split text into sentence-level claims for verification."""
    sentences = _SENTENCE_RE.split(text.strip())
    claims: list[str] = []
    for s in sentences:
        s = s.strip()
        if len(s) > 20:  # ignore tiny fragments
            claims.append(s)
        if len(claims) >= max_claims:
            break
    return claims if claims else [text[:500]]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class OmniTrustPipeline:
    """End-to-end OmniTrust RAG pipeline."""

    def __init__(self) -> None:
        self.bus = MessageBus()
        self.planner = PlannerAgent()
        self.strategist = StrategistAgent()
        self.critic = CriticAgent()
        self.synthesizer = SynthesizerAgent()
        self.retriever = HybridRetriever()
        self.independence_scorer = IndependenceScorer()
        self.utility_scorer = UtilityScorer()
        self.verifier = FamilyAttentionVerifier()

    def run(self, question: str) -> PipelineResponse:
        """Execute the full 10-step pipeline and return a ``PipelineResponse``."""
        self.bus.clear()
        metrics: dict[str, Any] = {}
        pipeline_t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Step 1 – Plan search queries
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        queries = self.planner.plan_queries(question, self.bus)
        metrics["step1_plan_queries_s"] = round(time.perf_counter() - t0, 3)
        logger.info("Step 1 complete: %d queries planned", len(queries))

        # ------------------------------------------------------------------
        # Step 2 – Strategist audit
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        audit = self.strategist.audit_plan(question, queries, self.bus)
        final_queries: list[str] = audit.get("final_queries", queries)
        strategist_decision: str = audit.get("decision", "approve")
        metrics["step2_strategist_audit_s"] = round(time.perf_counter() - t0, 3)
        logger.info("Step 2 complete: decision=%s, %d queries", strategist_decision, len(final_queries))

        # ------------------------------------------------------------------
        # Step 3 – Retrieve documents
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        all_docs: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for q in final_queries:
            for doc in self.retriever.search(q, top_k=5):
                if doc["id"] not in seen_ids:
                    seen_ids.add(doc["id"])
                    all_docs.append(doc)
        metrics["step3_retrieval_s"] = round(time.perf_counter() - t0, 3)
        metrics["step3_docs_retrieved"] = len(all_docs)
        self.bus.post("Pipeline", "documents_retrieved", {"count": len(all_docs)})
        logger.info("Step 3 complete: %d unique documents retrieved", len(all_docs))

        # ------------------------------------------------------------------
        # Step 4 – Score independence
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        doc_texts = [d.get("text", "") for d in all_docs]
        independence_scores = self.independence_scorer.score(doc_texts)
        metrics["step4_independence_scoring_s"] = round(time.perf_counter() - t0, 3)

        # ------------------------------------------------------------------
        # Step 5 – Score utility
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        utility_scores = self.utility_scorer.score(question, doc_texts)
        metrics["step5_utility_scoring_s"] = round(time.perf_counter() - t0, 3)

        # ------------------------------------------------------------------
        # Step 6 – Filter and annotate documents
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        document_results: list[DocumentResult] = []
        filtered_docs: list[dict[str, Any]] = []  # docs that pass filters

        for idx, doc in enumerate(all_docs):
            ind = independence_scores[idx] if idx < len(independence_scores) else {}
            utl = utility_scores[idx] if idx < len(utility_scores) else {}

            is_dup = ind.get("is_duplicate", False)
            is_useful = utl.get("is_useful", True)

            dr = DocumentResult(
                id=doc.get("id", str(idx)),
                title=doc.get("title", ""),
                text=doc.get("text", ""),
                source=doc.get("source", ""),
                independence_score=ind.get("independence_score", 1.0),
                utility_score=utl.get("utility_score", 0.5),
                novelty=utl.get("novelty", 0.5),
                contradiction=utl.get("contradiction", 0.0),
                is_duplicate=is_dup,
                is_useful=is_useful,
            )
            document_results.append(dr)

            # Keep non-duplicate, useful docs for downstream
            if not is_dup and is_useful:
                filtered_docs.append(doc)

        # If filtering removed everything, fall back to all docs
        if not filtered_docs and all_docs:
            filtered_docs = all_docs

        metrics["step6_filtering_s"] = round(time.perf_counter() - t0, 3)
        metrics["step6_docs_after_filter"] = len(filtered_docs)
        self.bus.post("Pipeline", "documents_filtered", {"kept": len(filtered_docs), "total": len(all_docs)})
        logger.info("Step 6 complete: %d/%d docs passed filters", len(filtered_docs), len(all_docs))

        # ------------------------------------------------------------------
        # Step 7 – Family-attention verification
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        # Extract preliminary claims from the top evidence for verification
        evidence_text = " ".join(d.get("text", "")[:500] for d in filtered_docs[:5])
        preliminary_claims = _extract_claims(evidence_text)
        ver_raw = self.verifier.verify(preliminary_claims, filtered_docs)

        verification = VerificationResult(
            consistency_score=ver_raw.get("consistency_score", 0.0),
            js_divergence=ver_raw.get("js_divergence", 0.0),
            head_perspectives=[
                HeadPerspective(**hp) for hp in ver_raw.get("head_perspectives", [])
            ],
            similarity_matrix=ver_raw.get("similarity_matrix", []),
        )
        metrics["step7_verification_s"] = round(time.perf_counter() - t0, 3)
        self.bus.post("Pipeline", "verification_complete", {"consistency": verification.consistency_score})
        logger.info("Step 7 complete: consistency=%.3f", verification.consistency_score)

        # ------------------------------------------------------------------
        # Step 8 – Critic checks claims
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        evidence_for_critic = [
            {"id": d.get("id", ""), "title": d.get("title", ""), "text": d.get("text", "")}
            for d in filtered_docs[:4]
        ]
        checked_raw = self.critic.check_claims(preliminary_claims, evidence_for_critic, self.bus)
        checked_claims = [
            CheckedClaim(
                claim=c.get("claim", ""),
                status=c.get("status", "uncertain"),
                evidence_doc_id=c.get("evidence_doc_id", ""),
                evidence_chunk_id=c.get("evidence_chunk_id", ""),
                reason=c.get("reason", ""),
            )
            for c in checked_raw
        ]
        metrics["step8_critic_s"] = round(time.perf_counter() - t0, 3)
        logger.info("Step 8 complete: %d claims checked", len(checked_claims))

        # ------------------------------------------------------------------
        # Step 9 – Synthesise final answer
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        final_answer = self.synthesizer.synthesize(
            question=question,
            verified_claims=checked_raw,
            evidence=evidence_for_critic,
            bus=self.bus,
        )
        metrics["step9_synthesis_s"] = round(time.perf_counter() - t0, 3)
        logger.info("Step 9 complete: answer length=%d chars", len(final_answer))

        # ------------------------------------------------------------------
        # Step 10 – Assemble response
        # ------------------------------------------------------------------
        metrics["total_pipeline_s"] = round(time.perf_counter() - pipeline_t0, 3)

        agent_logs = [
            AgentLog(
                sender=m.sender,
                msg_type=m.msg_type,
                payload=m.payload,
                timestamp=m.timestamp,
            )
            for m in self.bus.read()
        ]

        response = PipelineResponse(
            question=question,
            queries_used=final_queries,
            strategist_decision=strategist_decision,
            documents=document_results,
            verification=verification,
            checked_claims=checked_claims,
            final_answer=final_answer,
            agent_logs=agent_logs,
            pipeline_metrics=metrics,
        )
        logger.info("Pipeline complete in %.2fs", metrics["total_pipeline_s"])
        return response
