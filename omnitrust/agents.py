"""
omnitrust/agents.py — Multi-agent components for OmniTrust-RAG.
"""

from __future__ import annotations
import os
import json
from typing import Any, List, Dict
from openai import OpenAI
from omnitrust.bus import MessageBus

class BaseAgent:
    """Base class for OmniTrust RAG agents."""
    def __init__(self, role: str, model_id: str = "nvidia/llama-3.1-nemotron-nano-8b-v1"):
        self.role = role
        self.model_id = model_id
        self.api_key = os.getenv("NVIDIA_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        self.client = None
        if self.api_key:
            # NVIDIA NIM exposes OpenAI compatible API
            base_url = "https://integrate.api.nvidia.com/v1" if "nvidia" in model_id or os.getenv("NVIDIA_API_KEY") else None
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        """Call LLM with query and return text."""
        if not self.client:
            # Mock fallback if no API keys are provided
            return self._mock_generate(prompt)

        try:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.2,
                max_tokens=1024
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"[{self.role}] LLM call error: {e}")
            return self._mock_generate(prompt)

    def _mock_generate(self, prompt: str) -> str:
        """Fallback mock generator for local testing."""
        prompt_lower = prompt.lower()
        if "query" in prompt_lower or "search" in prompt_lower:
            return json.dumps({
                "queries": [
                    "transformer architecture detail and components",
                    "multi-head self-attention mechanism",
                    "attention is all you need paper summary"
                ]
            })
        elif "audit" in prompt_lower:
            return json.dumps({
                "decision": "approve",
                "reason": "The selected search strategy is highly relevant.",
                "remove_chunk_ids": [],
                "priority_boost": ["C0", "C1"]
            })
        elif "critic" in prompt_lower or "evidence" in prompt_lower:
            return json.dumps({
                "checked_claims": [
                    {
                        "claim": "Transformer architecture uses self-attention.",
                        "status": "supported",
                        "evidence_doc_id": "doc_0",
                        "evidence_chunk_id": "C0",
                        "reason": "Directly supported by the text."
                    }
                ]
            })
        else:
            return "Based on the retrieved documents, the system confirms the facts are verified and supported by the sources."


class PlannerAgent(BaseAgent):
    """Planner Agent: Generates search queries optimized for finding relevant evidence."""
    def __init__(self):
        super().__init__(role="Planner", model_id="nvidia/llama-3.1-nemotron-nano-8b-v1")

    def plan_queries(self, question: str, bus: MessageBus) -> List[str]:
        prompt = f"""You are the Planner Agent. Your job is to break down the user's question into 3 distinct search queries.
USER QUESTION: {question}

Respond ONLY with valid JSON in this format:
{{
  "queries": ["query 1", "query 2", "query 3"]
}}
"""
        response = self.generate(prompt)
        try:
            # Clean JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            data = json.loads(response.strip())
            queries = data.get("queries", [question])
        except Exception:
            queries = [question, f"{question} details", f"{question} summary"]

        bus.post(sender=self.role, msg_type="queries_planned", payload={"queries": queries})
        return queries


class StrategistAgent(BaseAgent):
    """Strategist Agent: Audits planning decisions and ensures retrieval scope matches intent."""
    def __init__(self):
        super().__init__(role="Strategist", model_id="nvidia/llama-3.3-nemotron-super-49b-v1")

    def audit_plan(self, question: str, queries: List[str], bus: MessageBus) -> Dict[str, Any]:
        prompt = f"""You are the Strategist Agent. Audit the queries proposed by the Planner for the user's question.

USER QUESTION: {question}
PLANNED QUERIES: {queries}

Respond ONLY with valid JSON in this format:
{{
  "decision": "approve|modify",
  "reason": "Brief explanation",
  "queries": ["query 1", "query 2"]
}}
"""
        response = self.generate(prompt)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            data = json.loads(response.strip())
        except Exception:
            data = {"decision": "approve", "reason": "No revision needed", "queries": queries}

        bus.post(sender=self.role, msg_type="plan_audited", payload=data)
        return data


class CriticAgent(BaseAgent):
    """Critic Agent: Audits synthesized claims and cross-checks them against source evidence."""
    def __init__(self):
        super().__init__(role="Critic", model_id="nvidia/llama-3.1-nemotron-nano-8b-v1")

    def check_claims(self, claims: List[str], evidence: List[Dict[str, Any]], bus: MessageBus) -> List[Dict[str, Any]]:
        prompt = f"""You are the Critic Agent. Cross-check the key claims against the retrieved evidence.
For each claim, determine if it is:
- "supported": evidence directly supports the factual meaning.
- "uncertain": evidence is related but insufficient to fully confirm.
- "unsupported": no evidence supports it.

CLAIMS TO CHECK:
{json.dumps(claims, indent=2)}

AVAILABLE EVIDENCE:
{json.dumps(evidence, indent=2)}

Respond ONLY with valid JSON:
{{
  "checked_claims": [
    {{
      "claim": "claim text",
      "status": "supported|uncertain|unsupported",
      "evidence_doc_id": "doc_id",
      "evidence_chunk_id": "chunk_id",
      "reason": "explanation"
    }}
  ]
}}
"""
        response = self.generate(prompt)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            data = json.loads(response.strip())
            checked = data.get("checked_claims", [])
        except Exception:
            checked = [{"claim": c, "status": "uncertain", "reason": "Parsing failed"} for c in claims]

        bus.post(sender=self.role, msg_type="claims_checked", payload={"checked_claims": checked})
        return checked


class SynthesizerAgent(BaseAgent):
    """Synthesizer Agent: Fuses verified evidence and formulates the final robust response."""
    def __init__(self):
        super().__init__(role="Synthesizer", model_id="nvidia/llama-3.3-nemotron-super-49b-v1")

    def synthesize(self, question: str, verified_claims: List[Dict[str, Any]], bus: MessageBus) -> str:
        prompt = f"""You are the Synthesizer Agent. Fuse the verified evidence to provide a comprehensive, hallucination-free answer.
Ignore any claims marked as 'unsupported'. State clearly if any facts remain 'uncertain'.

USER QUESTION: {question}
VERIFIED EVIDENCE CLAIMS:
{json.dumps(verified_claims, indent=2)}

Provide the final structured response:
"""
        answer = self.generate(prompt)
        bus.post(sender=self.role, msg_type="final_synthesis", payload={"answer": answer})
        return answer
