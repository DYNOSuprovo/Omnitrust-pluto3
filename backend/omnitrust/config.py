"""
OmniTrust-RAG Configuration Module.

Centralises all API keys, model identifiers, and tuneable parameters.
Environment variables take precedence over the compiled-in defaults so that
production deployments can inject secrets without touching source code.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

# Load local .env variables if present
load_dotenv()

# ---------------------------------------------------------------------------
# API Keys – overridable via environment / local .env file
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv(
    "GROQ_API_KEY",
    "",
)

NVIDIA_API_KEY: str = os.getenv(
    "NVIDIA_API_KEY",
    "",
)

# ---------------------------------------------------------------------------
# Groq Model Identifiers
# ---------------------------------------------------------------------------
GROQ_FAST_MODEL: str = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
GROQ_REASONING_MODEL: str = os.getenv("GROQ_REASONING_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# NVIDIA NIM Model Identifiers
# ---------------------------------------------------------------------------
NVIDIA_EMBED_MODEL: str = os.getenv(
    "NVIDIA_EMBED_MODEL", "nvidia/llama-nemotron-embed-1b-v2"
)
NVIDIA_RERANK_MODEL: str = os.getenv(
    "NVIDIA_RERANK_MODEL", "nvidia/rerank-qa-mistral-4b"
)

# ---------------------------------------------------------------------------
# NVIDIA NIM Endpoints
# ---------------------------------------------------------------------------
NVIDIA_RANKING_URL: str = os.getenv(
    "NVIDIA_RANKING_URL", "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
)

# ---------------------------------------------------------------------------
# Pipeline Tunables
# ---------------------------------------------------------------------------
RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "10"))
INDEPENDENCE_SIMILARITY_THRESHOLD: float = float(
    os.getenv("INDEPENDENCE_SIMILARITY_THRESHOLD", "0.85")
)
MAX_LLM_RETRIES: int = int(os.getenv("MAX_LLM_RETRIES", "2"))
LLM_RETRY_DELAY_SECONDS: float = float(os.getenv("LLM_RETRY_DELAY_SECONDS", "2.0"))
