from __future__ import annotations

import pytest

from winnow.exceptions import ModelDeclinedError, ParseFailedError
from winnow.parser.base import Parser


class SimpleParser(Parser[str]):
    """Test parser that returns the stripped response."""

    def parse(self, response: str) -> str:
        """Return stripped response."""
        return response.strip()


class TestParserDeclineDetection:
    def test_raises_for_exact_decline_keyword(self) -> None:
        """Verify parser raises ModelDeclinedError for exact DECLINE response."""
        parser = SimpleParser()

        with pytest.raises(ModelDeclinedError):
            parser(response="DECLINE")

    def test_raises_for_mixed_case_keyword(self) -> None:
        """Verify decline detection is case-insensitive."""
        parser = SimpleParser()

        with pytest.raises(ModelDeclinedError):
            parser(response="Decline")

    def test_raises_for_keyword_with_whitespace(self) -> None:
        """Verify decline detection strips whitespace."""
        parser = SimpleParser()

        with pytest.raises(ModelDeclinedError):
            parser(response="  DECLINE  ")

    def test_parses_when_keyword_embedded_in_text(self) -> None:
        """Verify parser does not decline when keyword is part of larger response."""
        parser = SimpleParser()

        result = parser(response="I DECLINE to answer")

        assert result == "I DECLINE to answer"

    def test_parses_normal_response(self) -> None:
        """Verify parser returns parsed value for normal responses."""
        parser = SimpleParser()

        result = parser(response="  hello world  ")

        assert result == "hello world"


class TestParseFailedError:
    def test_parse_failed_error_attributes(self) -> None:
        """Verify ParseFailedError stores response and reason."""
        error = ParseFailedError(response="test", reason="failed")

        assert error.response == "test"
        assert error.reason == "failed"


class TestModelDeclinedError:
    def test_model_declined_error_attributes(self) -> None:
        """Verify ModelDeclinedError stores response."""
        error = ModelDeclinedError(response="I cannot answer")

        assert error.response == "I cannot answer"
