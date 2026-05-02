"""LLM client protocol and OpenAI implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from openai import AsyncOpenAI


ReasoningEffort = Literal["minimal", "low", "medium", "high"]


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for sending a prompt to an LLM and receiving a response."""

    async def __call__(self, prompt: str) -> str: ...


class OpenAILLMClient:
    """LLM client backed by the OpenAI chat completions API."""

    def __init__(
        self,
        *,
        openai_client: AsyncOpenAI,
        model: str,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        self._openai_client = openai_client
        self._model = model
        self._reasoning_effort: ReasoningEffort | None = reasoning_effort

    async def __call__(self, prompt: str) -> str:
        """Send *prompt* to the configured model and return the response."""
        if self._reasoning_effort is not None:
            response = await self._openai_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=self._reasoning_effort,
            )
        else:
            response = await self._openai_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content
