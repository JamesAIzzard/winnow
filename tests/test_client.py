from __future__ import annotations

from typing import Any, Iterator, cast

import pytest
from openai import APIConnectionError

from winnow.client import _ParsedOpenAIClient
from winnow.exceptions import LLMRetriesError
from winnow.parsers import parse_string


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.output_text = text


class TestEnsureConnectionErrors:
    def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify ensure retries connection errors then returns parsed result."""
        client = _ParsedOpenAIClient(
            model="test-model",
            max_retries=3,
            allow_decline=False,
        )

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setattr(
            "openai_parsed.client.random.uniform",
            lambda *_args, **_kwargs: 0.0,
        )
        sleep_calls: list[float] = []
        monkeypatch.setattr(
            "openai_parsed.client.time.sleep",
            sleep_calls.append,
        )

        responses: Iterator[object] = iter(
            [
                APIConnectionError(
                    request=cast(Any, None), message="temporary issue 1"
                ),
                APIConnectionError(
                    request=cast(Any, None), message="temporary issue 2"
                ),
                FakeResponse("ok"),
            ]
        )

        def fake_create(*, model: str, input: str) -> FakeResponse:
            result = next(responses)
            if isinstance(result, Exception):
                raise result
            return result  # type: ignore[return-value]

        monkeypatch.setattr("openai.responses.create", fake_create)

        result = client.ensure(prompt="hello", parser=parse_string)

        assert result == "ok"
        assert sleep_calls == [0.5, 1.0]

    def test_exhausts_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify ensure raises LLMRetriesError after exhausting connection retries."""
        client = _ParsedOpenAIClient(
            model="test-model",
            max_retries=2,
            allow_decline=False,
        )

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setattr(
            "openai_parsed.client.random.uniform",
            lambda *_args, **_kwargs: 0.0,
        )
        sleep_calls: list[float] = []
        monkeypatch.setattr(
            "openai_parsed.client.time.sleep",
            sleep_calls.append,
        )

        def fake_create(*, model: str, input: str) -> FakeResponse:
            raise APIConnectionError(
                request=cast(Any, None), message="temporary issue"
            )

        monkeypatch.setattr("openai.responses.create", fake_create)

        with pytest.raises(LLMRetriesError) as error:
            client.ensure(prompt="hi", parser=parse_string)

        assert error.value.max_retries == 2
        assert sleep_calls == [0.5]
        assert error.value.response_log[1].startswith("APIConnectionError:")
