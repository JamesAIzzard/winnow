from __future__ import annotations

import pytest

from winnow.exceptions import ModelDeclinedError, ParseFailedError
from winnow.parser.numerical import FloatParser


class TestFloatParser:
    def test_parses_integer(self) -> None:
        """Verify parser handles integer values."""
        parser = FloatParser()

        result = parser(response="42")

        assert result == 42.0

    def test_parses_decimal(self) -> None:
        """Verify parser handles decimal values."""
        parser = FloatParser()

        result = parser(response="3.14159")

        assert result == pytest.approx(3.14159)

    def test_parses_negative_number(self) -> None:
        """Verify parser handles negative values."""
        parser = FloatParser()

        result = parser(response="-5.5")

        assert result == -5.5

    def test_parses_number_with_whitespace(self) -> None:
        """Verify parser strips whitespace."""
        parser = FloatParser()

        result = parser(response="  31.5  ")

        assert result == 31.5

    def test_raises_for_non_numeric(self) -> None:
        """Verify parser raises ParseFailedError for non-numeric input."""
        parser = FloatParser()

        with pytest.raises(ParseFailedError) as exc_info:
            parser(response="hello")

        assert exc_info.value.response == "hello"

    def test_raises_for_decline(self) -> None:
        """Verify parser raises ModelDeclinedError for decline keywords."""
        parser = FloatParser()

        with pytest.raises(ModelDeclinedError):
            parser(response="DECLINE")
