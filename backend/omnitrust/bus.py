"""
OmniTrust-RAG Message Bus.

Append-only, in-process event log that every agent writes to and reads from.
Designed for single-pipeline invocations; call ``clear()`` between runs if the
bus instance is reused.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    """Immutable message posted to the bus."""

    sender: str
    msg_type: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MessageBus:
    """Thread-safe, append-only message log for inter-agent communication."""

    def __init__(self) -> None:
        self._log: list[Message] = []
        self._lock = threading.Lock()

    # ---- write ----------------------------------------------------------

    def post(self, sender: str, msg_type: str, payload: dict[str, Any]) -> Message:
        """Append a new message and return it."""
        msg = Message(sender=sender, msg_type=msg_type, payload=payload)
        with self._lock:
            self._log.append(msg)
        return msg

    # ---- read -----------------------------------------------------------

    def read(self, *, sender: str | None = None, msg_type: str | None = None) -> list[Message]:
        """Return messages matching the optional filters (AND logic)."""
        with self._lock:
            result = list(self._log)
        if sender is not None:
            result = [m for m in result if m.sender == sender]
        if msg_type is not None:
            result = [m for m in result if m.msg_type == msg_type]
        return result

    def latest(self, *, sender: str | None = None, msg_type: str | None = None) -> Message | None:
        """Return the most recent matching message, or ``None``."""
        matches = self.read(sender=sender, msg_type=msg_type)
        return matches[-1] if matches else None

    # ---- introspection --------------------------------------------------

    def dump(self) -> list[dict[str, Any]]:
        """Serialise the full log to a list of dicts."""
        with self._lock:
            return [m.to_dict() for m in self._log]

    def summary(self) -> str:
        """Return a human-readable summary of every message on the bus."""
        with self._lock:
            messages = list(self._log)
        if not messages:
            return "Message bus is empty."

        lines: list[str] = [f"Message Bus – {len(messages)} message(s):"]
        for idx, msg in enumerate(messages, start=1):
            ts = time.strftime("%H:%M:%S", time.localtime(msg.timestamp))
            payload_keys = ", ".join(msg.payload.keys()) if msg.payload else "(empty)"
            lines.append(
                f"  [{idx}] {ts}  {msg.sender:>16s} | {msg.msg_type:<24s} | keys: {payload_keys}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        with self._lock:
            return len(self._log)

    # ---- lifecycle ------------------------------------------------------

    def clear(self) -> None:
        """Wipe the log for a fresh pipeline run."""
        with self._lock:
            self._log.clear()
