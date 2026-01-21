from __future__ import annotations

import pytest

from winnow.exceptions import ModelDeclinedError, ParseFailedError
from winnow.parser.optional_bounded import OptionalBoundedIntParser


class TestOptionalBoundedIntParser:

    def test_parses_minimum_value(self) -> None:
        """Verify parser accepts minimum bound value."""
        parser = OptionalBoundedIntParser(min_value=0, max_value=100)

        result = parser(response="0")

        assert result == 0

    def test_parses_maximum_value(self) -> None:
        """Verify parser accepts maximum bound value."""
        parser = OptionalBoundedIntParser(min_value=0, max_value=100)

        result = parser(response="100")

        assert result == 100

    def test_parses_none_lowercase(self) -> None:
        """Verify parser returns None for 'none'."""
        parser = OptionalBoundedIntParser(min_value=0, max_value=100)

        result = parser(response="none")

        assert result is None

    def test_parses_none_uppercase(self) -> None:
        """Verify parser returns None for 'None'."""
        parser = OptionalBoundedIntParser(min_value=0, max_value=100)

        result = parser(response="None")

        assert result is None

    def test_parses_number_with_whitespace(self) -> None:
        """Verify parser strips whitespace."""
        parser = OptionalBoundedIntParser(min_value=0, max_value=100)

        result = parser(response="  55  ")

        assert result == 55

    def test_raises_for_below_minimum(self) -> None:
        """Verify parser raises ParseFailedError for value below minimum."""
        parser = OptionalBoundedIntParser(min_value=0, max_value=100)

        with pytest.raises(ParseFailedError) as exc_info:
            parser(response="-5")

        assert exc_info.value.response == "-5"

    def test_raises_for_above_maximum(self) -> None:
        """Verify parser raises ParseFailedError for value above maximum."""
        parser = OptionalBoundedIntParser(min_value=0, max_value=100)

        with pytest.raises(ParseFailedError) as exc_info:
            parser(response="150")

        assert exc_info.value.response == "150"

    def test_raises_for_non_numeric(self) -> None:
        """Verify parser raises ParseFailedError for non-numeric, non-none input."""
        parser = OptionalBoundedIntParser(min_value=0, max_value=100)

        with pytest.raises(ParseFailedError) as exc_info:
            parser(response="hello")

        assert exc_info.value.response == "hello"

    def test_raises_for_decline(self) -> None:
        """Verify parser raises ModelDeclinedError for decline keywords."""
        parser = OptionalBoundedIntParser(min_value=0, max_value=100)

        with pytest.raises(ModelDeclinedError):
            parser(response="DECLINE")

    def test_custom_bounds(self) -> None:
        """Verify parser respects custom bounds."""
        parser = OptionalBoundedIntParser(min_value=10, max_value=50)

        assert parser(response="25") == 25

        with pytest.raises(ParseFailedError):
            parser(response="5")

        with pytest.raises(ParseFailedError):
            parser(response="75")
