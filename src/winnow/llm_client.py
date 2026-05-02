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
    """LLM client backed by the OpenAI API.

    Plain calls go through chat.completions. When *web_search* is set on a
    call, the request is routed through the Responses API with the
    ``web_search`` tool attached so the model can ground its answer on
    live results.
    """

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

    async def __call__(self, prompt: str, *, web_search: bool = False) -> str:
        """Send *prompt* to the configured model and return the response."""
        if web_search:
            return await self._respond_with_web_search(prompt)
        return await self._respond(prompt)

    async def _respond(self, prompt: str) -> str:
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

    async def _respond_with_web_search(self, prompt: str) -> str:
        if self._reasoning_effort is not None:
            response = await self._openai_client.responses.create(
                model=self._model,
                input=prompt,
                tools=[{"type": "web_search"}],
                reasoning={"effort": self._reasoning_effort},
            )
        else:
            response = await self._openai_client.responses.create(
                model=self._model,
                input=prompt,
                tools=[{"type": "web_search"}],
            )
        return response.output_text or ""
