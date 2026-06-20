"""
OmniTrust-RAG Multi-Agent System.

Each agent wraps the Groq chat-completion API and writes its actions to the
shared ``MessageBus`` so every step is auditable.  JSON parsing from LLM
output uses a regex-based fallback to tolerate markdown code fences and
conversational preamble.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from groq import Groq

from omnitrust.bus import MessageBus
from omnitrust.config import (
    GROQ_API_KEY,
    GROQ_FAST_MODEL,
    GROQ_REASONING_MODEL,
    LLM_RETRY_DELAY_SECONDS,
    MAX_LLM_RETRIES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"(\[.*\])", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _extract_json(text: str) -> Any:
    """Best-effort JSON extraction from LLM output.

    Strategy:
    1. Try to parse the whole text directly.
    2. Look for a fenced ```json ... ``` block.
    3. Look for the first top-level JSON array.
    4. Look for the first top-level JSON object.
    5. Give up and return the raw text.
    """
    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Fenced code block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Top-level array
    m = _JSON_ARRAY_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. Top-level object
    m = _JSON_OBJECT_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # 5. Fallback – return raw text
    return text


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------


class BaseAgent:
    """Thin wrapper around the Groq chat-completion API with retries."""

    def __init__(self, role: str, model_id: str) -> None:
        self.role = role
        self.model_id = model_id
        self._client = Groq(api_key=GROQ_API_KEY)

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        """Call the Groq API with retry logic and return the assistant text."""
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        for attempt in range(1, MAX_LLM_RETRIES + 2):  # attempt 1 … MAX+1
            try:
                t0 = time.perf_counter()
                response = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=4096,
                )
                elapsed = time.perf_counter() - t0
                text = response.choices[0].message.content or ""
                logger.info(
                    "[%s] Groq %s responded in %.2fs (%d chars)",
                    self.role,
                    self.model_id,
                    elapsed,
                    len(text),
                )
                return text
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "[%s] Groq call attempt %d/%d failed: %s",
                    self.role,
                    attempt,
                    MAX_LLM_RETRIES + 1,
                    exc,
                )
                if attempt <= MAX_LLM_RETRIES:
                    time.sleep(LLM_RETRY_DELAY_SECONDS)

        # Fallback to GROQ_FAST_MODEL if reasoning model fails (e.g. on 429 rate limit)
        if self.model_id == GROQ_REASONING_MODEL and GROQ_FAST_MODEL != GROQ_REASONING_MODEL:
            logger.warning(
                "[%s] Reasoning model %s failed (possibly due to rate limits). "
                "Attempting fallback to fast model: %s",
                self.role,
                self.model_id,
                GROQ_FAST_MODEL,
            )
            try:
                t0 = time.perf_counter()
                response = self._client.chat.completions.create(
                    model=GROQ_FAST_MODEL,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=4096,
                )
                elapsed = time.perf_counter() - t0
                text = response.choices[0].message.content or ""
                logger.info(
                    "[%s] Groq fallback %s responded in %.2fs (%d chars)",
                    self.role,
                    GROQ_FAST_MODEL,
                    elapsed,
                    len(text),
                )
                return text
            except Exception as fallback_exc:
                logger.exception("[%s] Fallback to fast model failed", self.role)
                raise RuntimeError(
                    f"[{self.role}] All API attempts on {self.model_id} failed. "
                    f"Fallback to {GROQ_FAST_MODEL} also failed. Last error: {fallback_exc}"
                ) from fallback_exc

        raise RuntimeError(
            f"[{self.role}] All {MAX_LLM_RETRIES + 1} Groq API attempts failed. "
            f"Last error: {last_error}"
        )


# ---------------------------------------------------------------------------
# Planner Agent
# ---------------------------------------------------------------------------


class PlannerAgent(BaseAgent):
    """Generates optimised search queries for a user question."""

    def __init__(self) -> None:
        super().__init__(role="Planner", model_id=GROQ_FAST_MODEL)

    def plan_queries(self, question: str, bus: MessageBus) -> list[str]:
        system = (
            "You are a search-query planner for a retrieval-augmented generation system. "
            "Given a user question, produce 3 to 5 diverse, optimised search queries that "
            "together will cover the key facets of the question. Each query should target "
            "different aspects, perspectives, or related sub-topics.\n\n"
            "CRITICAL: Keep your queries tightly focused on the core subject of the question. "
            "Do NOT expand queries to general categories, unrelated topics, or tangential domains. "
            "For example, if the question is about 'coffee', every query must contain the word 'coffee' "
            "or direct terms like 'caffeine'; do NOT generate queries about 'stimulants', 'e-cigarettes', "
            "or unrelated health issues/substances.\n\n"
            "Return ONLY a JSON array of strings, e.g.:\n"
            '["query one", "query two", "query three"]'
        )
        prompt = f"User question:\n{question}"
        raw = self.generate(prompt, system)
        parsed = _extract_json(raw)

        if isinstance(parsed, list):
            queries = [str(q) for q in parsed if isinstance(q, str) and q.strip()]
        else:
            # Fallback: split by newlines and strip numbering
            queries = [
                re.sub(r"^\d+[\.\)]\s*", "", line).strip()
                for line in str(parsed).splitlines()
                if line.strip()
            ]

        # Guarantee at least the original question
        if not queries:
            queries = [question]

        bus.post(
            sender=self.role,
            msg_type="queries_planned",
            payload={"question": question, "queries": queries},
        )
        return queries


# ---------------------------------------------------------------------------
# Strategist Agent
# ---------------------------------------------------------------------------


class StrategistAgent(BaseAgent):
    """Reviews and optionally refines the planner's query set."""

    def __init__(self) -> None:
        super().__init__(role="Strategist", model_id=GROQ_REASONING_MODEL)

    def audit_plan(
        self, question: str, queries: list[str], bus: MessageBus
    ) -> dict[str, Any]:
        system = (
            "You are a senior research strategist. You receive a user question and a list "
            "of proposed search queries produced by a planner agent. Your job is to:\n"
            "1. Decide whether the queries are sufficient (approve), need changes (modify), "
            "   or require additional queries (expand).\n"
            "2. Return a JSON object with exactly two keys:\n"
            '   - "decision": one of "approve", "modify", "expand"\n'
            '   - "final_queries": the definitive list of search queries (3-6 strings)\n\n'
            "CRITICAL: Ensure that all final_queries are tightly locked to the core topic "
            "of the question. Reject or modify queries that expand to general categories, unrelated subjects, "
            "or unrelated diagnostics/substances (e.g., do not expand a coffee query to e-cigarettes or colonoscopy). "
            "Every query should contain the main subject name explicitly.\n\n"
            "Return ONLY the JSON object."
        )
        prompt = (
            f"User question:\n{question}\n\n"
            f"Proposed queries:\n{json.dumps(queries, indent=2)}"
        )
        raw = self.generate(prompt, system)
        parsed = _extract_json(raw)

        if isinstance(parsed, dict) and "final_queries" in parsed:
            result = {
                "decision": parsed.get("decision", "approve"),
                "final_queries": [str(q) for q in parsed["final_queries"]],
            }
        else:
            result = {"decision": "approve", "final_queries": queries}

        bus.post(
            sender=self.role,
            msg_type="plan_audited",
            payload=result,
        )
        return result


# ---------------------------------------------------------------------------
# Critic Agent
# ---------------------------------------------------------------------------


class CriticAgent(BaseAgent):
    """Cross-checks extracted claims against retrieved evidence."""

    def __init__(self) -> None:
        super().__init__(role="Critic", model_id=GROQ_FAST_MODEL)

    def check_claims(
        self,
        claims: list[str],
        evidence: list[dict[str, str]],
        bus: MessageBus,
    ) -> list[dict[str, Any]]:
        # Keep total context small to avoid hitting Groq's low TPM limit (6000 TPM on free tier)
        evidence_block = "\n\n".join(
            f"[DOC {e.get('id', idx)}] {e.get('title', '')}\n{e.get('text', '')[:500]}"
            for idx, e in enumerate(evidence[:3])
        )

        system = (
            "You are a meticulous fact-checker. For every claim below, determine whether "
            "the provided evidence SUPPORTS, makes the claim UNCERTAIN, or is UNSUPPORTED.\n\n"
            "Return a JSON array of objects. Each object must have exactly these keys:\n"
            '  "claim"  – the original claim text\n'
            '  "status" – one of "supported", "uncertain", "unsupported"\n'
            '  "evidence_doc_id" – the document ID used as evidence (or empty string)\n'
            '  "evidence_chunk_id" – a chunk reference if applicable (or empty string)\n'
            '  "reason" – one-sentence explanation\n\n'
            "Return ONLY the JSON array."
        )
        prompt = (
            f"Claims to check:\n{json.dumps(claims, indent=2)}\n\n"
            f"Evidence documents:\n{evidence_block}"
        )
        raw = self.generate(prompt, system)
        parsed = _extract_json(raw)

        if isinstance(parsed, list):
            checked: list[dict[str, Any]] = []
            for item in parsed:
                if isinstance(item, dict):
                    checked.append(
                        {
                            "claim": item.get("claim", ""),
                            "status": item.get("status", "uncertain"),
                            "evidence_doc_id": item.get("evidence_doc_id", ""),
                            "evidence_chunk_id": item.get("evidence_chunk_id", ""),
                            "reason": item.get("reason", ""),
                        }
                    )
            if not checked:
                checked = [
                    {
                        "claim": c,
                        "status": "uncertain",
                        "evidence_doc_id": "",
                        "evidence_chunk_id": "",
                        "reason": "LLM output could not be parsed.",
                    }
                    for c in claims
                ]
        else:
            checked = [
                {
                    "claim": c,
                    "status": "uncertain",
                    "evidence_doc_id": "",
                    "evidence_chunk_id": "",
                    "reason": "LLM output could not be parsed.",
                }
                for c in claims
            ]

        bus.post(sender=self.role, msg_type="claims_checked", payload={"checked_claims": checked})
        return checked


# ---------------------------------------------------------------------------
# Synthesizer Agent
# ---------------------------------------------------------------------------


class SynthesizerAgent(BaseAgent):
    """Produces the final, well-cited answer from verified claims and evidence."""

    def __init__(self) -> None:
        super().__init__(role="Synthesizer", model_id=GROQ_REASONING_MODEL)

    def synthesize(
        self,
        question: str,
        supported_claims: list[dict[str, Any]],
        unsupported_claims: list[dict[str, Any]],
        evidence: list[dict[str, str]],
        bus: MessageBus,
    ) -> str:
        evidence_block = "\n\n".join(
            f"[DOC {e.get('id', idx)}] {e.get('title', '')}\n{e.get('text', '')[:600]}"
            for idx, e in enumerate(evidence[:3])
        )
        supported_block = "\n".join(
            f"- {c.get('claim', '')} (Source: [DOC {c.get('evidence_doc_id', 'unknown')}])"
            for c in supported_claims
        ) if supported_claims else "None"

        unsupported_block = "\n".join(
            f"- {c.get('claim', '')}"
            for c in unsupported_claims
        ) if unsupported_claims else "None"

        system = (
            "You are an expert research synthesiser. Using ONLY the provided evidence "
            "documents and the verified claims below, write a comprehensive, well-structured "
            "answer to the user's question.\n\n"
            "TEMPORAL RELEVANCE / CHRONOLOGY:\n"
            "• You MUST prioritize the most chronologically recent facts found in the evidence (look at dates, years, and references like 'today', 'now', or 'currently').\n"
            "• If there is a change or transition over time (e.g., who is the UK Prime Minister today), you MUST present the current/latest status as the fact and clearly state that previous statuses are outdated.\n"
            "• Rely STRICTLY on the retrieved evidence documents. Do NOT allow your static, pre-trained parametric knowledge (or outdated external knowledge) to override newer, explicit dates/facts in the retrieved documents.\n\n"
            "CLAIM FACT-CHECKING RULES:\n"
            "• Under 'Supported Claims', you are given claims that have been verified against the evidence. You may assert these as true.\n"
            "• Under 'Unsupported/Refuted Claims', you are given claims that were checked and found to be UNSUPPORTED or REFUTED. You MUST NOT assert them as facts. If these claims are central to the user's query, you must explicitly state that the evidence does NOT support them.\n\n"
            "GENERAL GUIDELINES:\n"
            "• Cite specific documents by their [DOC ...] identifier.\n"
            "• Clearly state where evidence is strong, uncertain, or missing.\n"
            "• Use headings, bullet points, and logical flow.\n"
            "• Do NOT invent or assume information beyond what the evidence supports.\n"
            "• If conflicting evidence exists, present both sides.\n"
        )
        prompt = (
            f"Question:\n{question}\n\n"
            f"Supported Claims (Allowed to use as facts):\n{supported_block}\n\n"
            f"Unsupported/Refuted Claims (Forbidden to use as facts):\n{unsupported_block}\n\n"
            f"Evidence Documents:\n{evidence_block}"
        )
        answer = self.generate(prompt, system)

        bus.post(
            sender=self.role,
            msg_type="final_synthesis",
            payload={"answer": answer},
        )
        return answer
