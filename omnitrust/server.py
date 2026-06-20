"""
omnitrust/server.py — FastAPI server and orchestrator for OmniTrust-RAG.
"""

from __future__ import annotations
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any

from omnitrust.bus import MessageBus
from omnitrust.agents import PlannerAgent, StrategistAgent, CriticAgent, SynthesizerAgent
from omnitrust.scorer import IndependenceScorer, UtilityScorer
from omnitrust.family_verifier import FamilyAttentionVerifier

app = FastAPI(title="OmniTrust-RAG Server", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock Corpus Document Store
CORPUS = [
    {
        "id": "doc_0",
        "title": "Transformer Architecture Introduction",
        "text": "The Transformer architecture was introduced in the landmark paper 'Attention Is All You Need' by Vaswani et al. in 2017. It replaced recurrence and convolutions with self-attention."
    },
    {
        "id": "doc_1",
        "title": "Self-Attention Mechanism",
        "text": "A Transformer model relies entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned RNNs or convolution."
    },
    {
        "id": "doc_2",
        "title": "Multi-Head Attention",
        "text": "The original Transformer model uses eight parallel attention heads. Multi-head attention allows the model to jointly attend to information from different representation subspaces."
    },
    {
        "id": "doc_3",
        "title": "Quantum Computing Principles",
        "text": "Quantum computing utilizes superposition and entanglement to perform calculations. Qubits can exist in multiple states simultaneously, allowing for parallel computations."
    },
    {
        "id": "doc_4",
        "title": "Quantum Entanglement",
        "text": "Entanglement in quantum systems is a physical phenomenon where pairs or groups of particles generate quantum states such that the state of each particle cannot be described independently."
    },
    {
        "id": "doc_5",
        "title": "Duplicate Self-Attention Article (Copied)",
        "text": "A Transformer architecture relies entirely on self-attention mechanisms to compute representations of its inputs and outputs without RNNs or convolution. (Copied from blog)"
    }
]

class QueryRequest(BaseModel):
    question: str

@app.get("/api/corpus")
def get_corpus():
    return CORPUS

@app.post("/api/query")
def run_query(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # 1. Initialize MessageBus and Agents
    bus = MessageBus()
    planner = PlannerAgent()
    strategist = StrategistAgent()
    critic = CriticAgent()
    synthesizer = SynthesizerAgent()
    
    independence_scorer = IndependenceScorer(use_ml=False)
    utility_scorer = UtilityScorer(use_ml=False)
    family_verifier = FamilyAttentionVerifier(num_heads=4)

    # 2. Planning Phase (Planner + Strategist Council)
    planned_queries = planner.plan_queries(question, bus)
    audit_verdict = strategist.audit_plan(question, planned_queries, bus)
    
    # Use approved/modified queries from strategist
    search_queries = audit_verdict.get("queries", planned_queries)

    # 3. Retrieval Phase
    retrieved_docs = []
    seen_ids = set()
    
    # Keyword search across mock corpus
    for q in search_queries:
        keywords = set(q.lower().split())
        for doc in CORPUS:
            if doc["id"] in seen_ids:
                continue
            doc_words = set(doc["text"].lower().split())
            overlap = len(keywords.intersection(doc_words))
            if overlap >= 2: # minimum overlap threshold
                retrieved_docs.append(doc)
                seen_ids.add(doc["id"])

    # Fallback to general documents if none matched
    if not retrieved_docs:
        retrieved_docs = CORPUS[:3]

    doc_texts = [d["text"] for d in retrieved_docs]

    # 4. Evidence Scoring (Independence + Utility)
    independence_results = independence_scorer.score_independence(doc_texts)
    utility_results = utility_scorer.score_utility(question, doc_texts)

    # Filter out duplicates and low utility documents
    filtered_docs = []
    filtered_texts = []
    for idx, doc in enumerate(retrieved_docs):
        is_dup = independence_results[idx]["is_duplicate"]
        is_useful = utility_results[idx]["is_useful"]
        if not is_dup and is_useful:
            filtered_docs.append(doc)
            filtered_texts.append(doc_texts[idx])

    # Ensure at least one document remains
    if not filtered_docs:
        filtered_docs = [retrieved_docs[0]]
        filtered_texts = [doc_texts[0]]

    # 5. Verification Phase (Family Attention Engine)
    claims_to_verify = [
        f"Claim {i+1}: {text[:60]}..."
        for i, text in enumerate(filtered_texts)
    ]
    verification_results = family_verifier.verify_statements(claims_to_verify, doc_texts)

    # 6. Critic & Synthesis Phase
    evidence_payload = [
        {"doc_id": d["id"], "text": d["text"]}
        for d in filtered_docs
    ]
    checked_claims = critic.check_claims(claims_to_verify, evidence_payload, bus)
    final_answer = synthesizer.synthesize(question, checked_claims, bus)

    # 7. Package and Return Results
    return {
        "question": question,
        "queries_used": search_queries,
        "strategist_decision": audit_verdict.get("decision", "approve"),
        "retrieved_documents": retrieved_docs,
        "filtered_documents": filtered_docs,
        "scores": {
            "independence": independence_results,
            "utility": utility_results,
        },
        "verification": verification_results,
        "checked_claims": checked_claims,
        "final_answer": final_answer,
        "agent_logs": bus.dump()
    }

# Serve static files if they exist
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("omnitrust.server:app", host="127.0.0.1", port=8000, reload=True)
