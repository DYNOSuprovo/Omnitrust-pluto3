"""
OmniTrust-RAG Pydantic Models.

Defines every request / response schema used by the FastAPI endpoints and
the internal pipeline so that validation and serialisation are consistent
throughout the codebase.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Incoming question from the front-end."""
    question: str = Field(..., min_length=1, description="The user's natural-language question.")


# ---------------------------------------------------------------------------
# Document-level results
# ---------------------------------------------------------------------------

class DocumentResult(BaseModel):
    """A single retrieved document after scoring."""
    id: str
    title: str
    text: str
    source: str
    independence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    utility_score: float = Field(default=0.5, ge=0.0, le=1.0)
    novelty: float = Field(default=0.5, ge=0.0, le=1.0)
    contradiction: float = Field(default=0.0, ge=0.0, le=1.0)
    is_duplicate: bool = False
    is_useful: bool = True


# ---------------------------------------------------------------------------
# Family-Attention Verification
# ---------------------------------------------------------------------------

class HeadPerspective(BaseModel):
    """Output from a single attention head in the verifier."""
    head_id: int
    name: str
    supported: float = Field(ge=0.0, le=1.0)
    uncertain: float = Field(ge=0.0, le=1.0)
    unsupported: float = Field(ge=0.0, le=1.0)


class VerificationResult(BaseModel):
    """Aggregated verification output across all heads."""
    consistency_score: float = Field(ge=0.0, le=1.0)
    js_divergence: float = Field(ge=0.0)
    head_perspectives: list[HeadPerspective] = Field(default_factory=list)
    similarity_matrix: list[list[float]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Claim checking
# ---------------------------------------------------------------------------

class CheckedClaim(BaseModel):
    """A single claim that has been verified against evidence."""
    claim: str
    status: str = Field(description="One of: supported, uncertain, unsupported")
    evidence_doc_id: str = ""
    evidence_chunk_id: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# Agent logging
# ---------------------------------------------------------------------------

class AgentLog(BaseModel):
    """A single entry from the message bus, exposed to the front-end."""
    sender: str
    msg_type: str
    payload: dict = Field(default_factory=dict)
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Corpus document for the POST /api/corpus endpoint
# ---------------------------------------------------------------------------

class CorpusDocument(BaseModel):
    """A document to be added to the in-memory corpus."""
    title: str
    text: str
    source: str = "user-upload"


# ---------------------------------------------------------------------------
# Full Pipeline Response
# ---------------------------------------------------------------------------

class PipelineResponse(BaseModel):
    """Everything the front-end needs to render a complete result page."""
    question: str
    queries_used: list[str] = Field(default_factory=list)
    strategist_decision: str = ""
    documents: list[DocumentResult] = Field(default_factory=list)
    verification: VerificationResult = Field(default_factory=VerificationResult)
    checked_claims: list[CheckedClaim] = Field(default_factory=list)
    final_answer: str = ""
    agent_logs: list[AgentLog] = Field(default_factory=list)
    pipeline_metrics: dict = Field(default_factory=dict)
