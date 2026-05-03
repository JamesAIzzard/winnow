"""LLM client protocol."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from winnow.question import Prompt


_logger = logging.getLogger("winnow")
T = TypeVar("T")


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for sending a prompt to an LLM and receiving a response."""

    async def __call__(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class LoggedLLMClient:
    """LLM client wrapper that records prompt/response exchanges."""

    query_fn: LLMClient

    async def query_prompt(self, prompt: Prompt[T]) -> str:
        """Query one prompt and log the resulting exchange."""
        prompt_body = prompt.build_prompt()
        response = await self.query_fn(prompt_body)
        log_exchange(uid=prompt.uid, prompt=prompt_body, response=response)
        return response


def log_exchange(*, uid: str, prompt: str, response: str) -> None:
    """Emit a JSONL record describing one prompt/response exchange."""
    record = json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question_uid": uid,
        "prompt": prompt,
        "response": response,
    })
    _logger.debug(record)
