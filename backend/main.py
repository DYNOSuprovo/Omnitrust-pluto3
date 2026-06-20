"""
OmniTrust-RAG – FastAPI Application Entry Point.

Exposes the full OmniTrust pipeline over a simple REST API:

* ``POST /api/query``   – run the full pipeline for a user question
* ``GET  /api/corpus``  – list documents in the in-memory corpus
* ``POST /api/corpus``  – add documents to the corpus
* ``GET  /api/health``  – liveness / readiness check

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from omnitrust.models import (
    CorpusDocument,
    PipelineResponse,
    QueryRequest,
)
from omnitrust.pipeline import OmniTrustPipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("omnitrust")

# ---------------------------------------------------------------------------
# Seed corpus – pre-loaded documents about AI / ML topics
# ---------------------------------------------------------------------------

_SEED_DOCUMENTS: list[dict[str, Any]] = [
    {
        "title": "Introduction to Machine Learning",
        "text": (
            "Machine learning is a branch of artificial intelligence that focuses on "
            "building systems that learn from data. Rather than being explicitly "
            "programmed, these systems identify patterns in data and make decisions "
            "with minimal human intervention. The three main types of machine learning "
            "are supervised learning, unsupervised learning, and reinforcement learning. "
            "Supervised learning uses labelled training data to learn a mapping from "
            "inputs to outputs. Unsupervised learning discovers hidden patterns in "
            "unlabelled data. Reinforcement learning trains agents to take actions in "
            "an environment to maximise cumulative reward."
        ),
        "source": "seed-corpus",
    },
    {
        "title": "Deep Learning and Neural Networks",
        "text": (
            "Deep learning is a subset of machine learning that uses neural networks "
            "with many layers (deep neural networks) to model complex patterns in data. "
            "Key architectures include convolutional neural networks (CNNs) for image "
            "recognition, recurrent neural networks (RNNs) for sequential data, and "
            "transformers for natural language processing. The transformer architecture, "
            "introduced in the 2017 paper 'Attention Is All You Need', revolutionised "
            "NLP and became the foundation for large language models like GPT and BERT. "
            "Training deep networks requires large datasets and significant compute "
            "resources, typically GPUs or TPUs."
        ),
        "source": "seed-corpus",
    },
    {
        "title": "Retrieval-Augmented Generation (RAG)",
        "text": (
            "Retrieval-Augmented Generation (RAG) is a technique that enhances large "
            "language models by grounding their outputs in external knowledge. Instead "
            "of relying solely on the model's parametric memory, RAG retrieves relevant "
            "documents from a knowledge base at inference time and includes them in the "
            "model's context window. This approach reduces hallucinations, improves "
            "factual accuracy, and allows the model to reference up-to-date information "
            "without retraining. Key components of a RAG system include a document store, "
            "a retriever (often based on dense embeddings or BM25), and a generator LLM."
        ),
        "source": "seed-corpus",
    },
    {
        "title": "Natural Language Processing Fundamentals",
        "text": (
            "Natural language processing (NLP) is the field of AI concerned with "
            "enabling computers to understand, interpret, and generate human language. "
            "Core NLP tasks include tokenisation, part-of-speech tagging, named entity "
            "recognition, sentiment analysis, machine translation, question answering, "
            "and text summarisation. Modern NLP is dominated by pre-trained language "
            "models that learn contextual representations of text from large corpora. "
            "Fine-tuning these models on task-specific data yields state-of-the-art "
            "results across many benchmarks."
        ),
        "source": "seed-corpus",
    },
    {
        "title": "AI Safety and Alignment",
        "text": (
            "AI safety research aims to ensure that artificial intelligence systems "
            "behave in ways that are beneficial and aligned with human values. Key "
            "challenges include the alignment problem (ensuring AI goals match human "
            "intentions), robustness (maintaining safe behaviour under distribution "
            "shift), interpretability (understanding why models make specific decisions), "
            "and scalable oversight (supervising increasingly capable systems). "
            "Techniques such as reinforcement learning from human feedback (RLHF), "
            "constitutional AI, and red-teaming are used to improve the safety of "
            "deployed models."
        ),
        "source": "seed-corpus",
    },
]

# ---------------------------------------------------------------------------
# Application lifespan – initialise pipeline + seed corpus
# ---------------------------------------------------------------------------

pipeline: OmniTrustPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Initialising OmniTrust-RAG pipeline…")
    pipeline = OmniTrustPipeline()
    pipeline.retriever.corpus.add_documents(_SEED_DOCUMENTS)
    logger.info(
        "Seed corpus loaded: %d documents", len(pipeline.retriever.corpus.documents)
    )
    yield
    logger.info("Shutting down OmniTrust-RAG pipeline.")
    pipeline = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OmniTrust-RAG API",
    description="Multi-agent retrieval-augmented generation with family-attention verification.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow the Vite dev server and common local origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_pipeline() -> OmniTrustPipeline:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised yet.")
    return pipeline


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health", tags=["system"])
async def health_check() -> dict[str, Any]:
    """Liveness / readiness probe."""
    p = _get_pipeline()
    return {
        "status": "healthy",
        "corpus_size": len(p.retriever.corpus.documents),
        "timestamp": time.time(),
    }


@app.post("/api/query", response_model=PipelineResponse, tags=["pipeline"])
async def run_query(req: QueryRequest) -> PipelineResponse:
    """Execute the full OmniTrust pipeline for a question."""
    p = _get_pipeline()
    logger.info("Received query: %s", req.question[:120])
    try:
        result = p.run(req.question)
        return result
    except Exception as exc:
        logger.exception("Pipeline error for question: %s", req.question[:120])
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/corpus", tags=["corpus"])
async def get_corpus() -> dict[str, Any]:
    """Return all documents currently in the in-memory corpus."""
    p = _get_pipeline()
    docs = p.retriever.corpus.documents
    return {"count": len(docs), "documents": docs}


@app.post("/api/corpus", tags=["corpus"])
async def add_corpus_documents(documents: list[CorpusDocument]) -> dict[str, Any]:
    """Add one or more documents to the in-memory corpus."""
    p = _get_pipeline()
    new_docs = [
        {"title": d.title, "text": d.text, "source": d.source} for d in documents
    ]
    p.retriever.corpus.add_documents(new_docs)
    return {
        "added": len(new_docs),
        "total": len(p.retriever.corpus.documents),
    }


# ---------------------------------------------------------------------------
# Serve Frontend Static Assets (Production Mode)
# ---------------------------------------------------------------------------
import os
from fastapi.staticfiles import StaticFiles

dist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(dist_path):
    logger.info("Serving production frontend from: %s", dist_path)
    app.mount("/", StaticFiles(directory=dist_path, html=True), name="frontend")
else:
    logger.warning("Production frontend build folder not found at: %s. Frontend will not be served from backend.", dist_path)


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
