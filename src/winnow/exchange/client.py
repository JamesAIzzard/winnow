"""Source-neutral query execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from ..progress import QuestionInteraction
from .logging import record_exchange

if TYPE_CHECKING:
    from ..question import Question


T = TypeVar("T")


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for sending a prompt to an LLM and receiving a response."""

    async def __call__(self, prompt: str) -> str: ...


@dataclass(frozen=True, kw_only=True)
class ExchangeRecordingClient:
    """Execute queries and record their completed exchanges."""

    query_fn: LLMClient

    async def query_question(self, question: Question[T]) -> QuestionInteraction:
        prompt_body = question.build_prompt()
        raw_response = await self.query_fn(prompt_body)
        interaction = QuestionInteraction(
            question_uid=question.uid,
            prompt=prompt_body,
            raw_response=raw_response,
        )
        record_exchange(
            uid=interaction.question_uid,
            prompt=interaction.prompt,
            response=interaction.raw_response,
        )
        return interaction
