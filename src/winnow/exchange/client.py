"""Source-neutral query execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from .logging import record_exchange

if TYPE_CHECKING:
    from ..question import Prompt


T = TypeVar("T")


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for sending a prompt to an LLM and receiving a response."""

    async def __call__(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class ExchangeRecordingClient:
    """Execute queries and record their completed exchanges."""

    query_fn: LLMClient

    async def query_prompt(self, prompt: Prompt[T]) -> str:
        prompt_body = prompt.build_prompt()
        response = await self.query_fn(prompt_body)
        record_exchange(uid=prompt.uid, prompt=prompt_body, response=response)
        return response
