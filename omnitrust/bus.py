"""
omnitrust/bus.py — Shared message bus for agent coordination.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Any

@dataclass
class Message:
    sender: str
    msg_type: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.perf_counter)

class MessageBus:
    """
    Append-only message bus shared across agents in a single pipeline run.
    """
    def __init__(self) -> None:
        self._messages: list[Message] = []

    def post(self, sender: str, msg_type: str, payload: dict[str, Any]) -> None:
        self._messages.append(Message(sender=sender, msg_type=msg_type, payload=payload))

    def read(self, msg_type: str | None = None, sender: str | None = None) -> list[Message]:
        msgs = self._messages
        if msg_type:
            msgs = [m for m in msgs if m.msg_type == msg_type]
        if sender:
            msgs = [m for m in msgs if m.sender == sender]
        return msgs

    def latest(self, msg_type: str) -> Message | None:
        matches = self.read(msg_type=msg_type)
        return matches[-1] if matches else None

    def dump(self) -> list[dict[str, Any]]:
        return [
            {
                "sender": m.sender,
                "type": m.msg_type,
                "payload": m.payload,
                "timestamp": round(m.timestamp, 4)
            }
            for m in self._messages
        ]
