from __future__ import annotations

import pytest

from winnow.exceptions import ModelDeclinedError, ParseFailedError
from winnow.parser.categorical import LiteralParser


class TestLiteralParser:
    def test_parses_valid_option(self) -> None:
        """Verify parser returns value for valid option."""
        parser: LiteralParser[str] = LiteralParser(
            frozenset({"apple", "banana", "cherry"})
        )

        result = parser(response="banana")

        assert result == "banana"

    def test_case_insensitive_by_default(self) -> None:
        """Verify parsing is case-insensitive by default."""
        parser: LiteralParser[str] = LiteralParser(
            frozenset({"apple", "banana", "cherry"})
        )

        result = parser(response="BANANA")

        assert result == "banana"

    def test_case_sensitive_when_specified(self) -> None:
        """Verify case-sensitive mode works."""
        parser: LiteralParser[str] = LiteralParser(
            frozenset({"Apple", "Banana", "Cherry"}),
            case_sensitive=True,
        )

        result = parser(response="Banana")
        assert result == "Banana"

        with pytest.raises(ParseFailedError):
            parser(response="banana")

    def test_strips_whitespace(self) -> None:
        """Verify whitespace is stripped."""
        parser: LiteralParser[str] = LiteralParser(
            frozenset({"apple", "banana", "cherry"})
        )

        result = parser(response="  banana  ")

        assert result == "banana"

    def test_raises_for_invalid_option(self) -> None:
        """Verify parser raises ParseFailedError for invalid option."""
        parser: LiteralParser[str] = LiteralParser(
            frozenset({"apple", "banana", "cherry"})
        )

        with pytest.raises(ParseFailedError) as exc_info:
            parser(response="mango")

        assert exc_info.value.response == "mango"

    def test_raises_for_decline(self) -> None:
        """Verify parser raises ModelDeclinedError for decline keywords."""
        parser: LiteralParser[str] = LiteralParser(
            frozenset({"apple", "banana", "cherry"})
        )

        with pytest.raises(ModelDeclinedError):
            parser(response="DECLINE")

    def test_works_with_non_string_options(self) -> None:
        """Verify parser works with non-string options."""
        parser: LiteralParser[int] = LiteralParser(frozenset({1, 2, 3}))

        result = parser(response="2")

        assert result == 2
