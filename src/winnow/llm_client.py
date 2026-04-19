"""LLM client protocol and OpenAI implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from openai import AsyncOpenAI


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for sending a prompt to an LLM and receiving a response."""

    async def __call__(self, prompt: str) -> str: ...


class OpenAILLMClient:
    """LLM client backed by the OpenAI chat completions API."""

    def __init__(self, *, openai_client: AsyncOpenAI, model: str) -> None:
        self._openai_client = openai_client
        self._model = model

    async def __call__(self, prompt: str) -> str:
        """Send *prompt* to the configured model and return the response."""
        response = await self._openai_client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content
