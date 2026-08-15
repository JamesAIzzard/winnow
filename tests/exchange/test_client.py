from __future__ import annotations

import asyncio
from typing import Any

import pytest

from winnow import QuestionInteraction
from winnow.estimator.numerical import NumericalEstimator
from winnow.exchange.client import ExchangeRecordingClient
from winnow.parser.numerical import FloatParser
from winnow.question import Question
from winnow.stopping import StoppingCriterion


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
        question = Question(
            uid="protein",
            query="prompt body",
            parser=FloatParser(),
            estimator=NumericalEstimator(),
            stopping_criterion=StoppingCriterion(),
        )

        interaction = asyncio.run(client.query_question(question))

        prompt_body = question.build_prompt()
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
