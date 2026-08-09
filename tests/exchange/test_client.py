from __future__ import annotations

import asyncio
from typing import Any

import pytest

from winnow import QuestionInteraction
from winnow.exchange.client import ExchangeRecordingClient
from winnow.parser.numerical import FloatParser
from winnow.question import Prompt


class TestExchangeRecordingClient:
    def test_builds_queries_and_records_completed_exchanges(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        events: list[tuple[str, Any]] = []

        async def query_fn(prompt: str) -> str:
            events.append(("query", prompt))
            return f"response to {prompt}"

        def record_exchange(**exchange: str) -> None:
            events.append(("record", exchange))

        monkeypatch.setattr(
            "winnow.exchange.client.record_exchange",
            record_exchange,
        )
        client = ExchangeRecordingClient(query_fn=query_fn)
        prompt = Prompt(uid="protein", query="prompt body", parser=FloatParser())

        interaction = asyncio.run(client.query_prompt(prompt))

        prompt_body = prompt.build_prompt()
        assert interaction == QuestionInteraction(
            question_uid="protein",
            prompt=prompt_body,
            raw_response=f"response to {prompt_body}",
        )
        assert events == [
            ("query", prompt_body),
            (
                "record",
                {
                    "uid": interaction.question_uid,
                    "prompt": interaction.prompt,
                    "response": interaction.raw_response,
                },
            ),
        ]
