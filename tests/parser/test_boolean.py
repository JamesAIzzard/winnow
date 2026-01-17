from __future__ import annotations

import pytest

from winnow.parser.base import ParserError
from winnow.parser.boolean import BooleanParser


class TestBooleanParser:
    def test_parses_yes_as_true(self) -> None:
        """Verify 'yes' parses to True."""
        parser = BooleanParser()

        result = parser(response="yes")

        assert result is True

    def test_parses_true_as_true(self) -> None:
        """Verify 'true' parses to True."""
        parser = BooleanParser()

        result = parser(response="true")

        assert result is True

    def test_parses_y_as_true(self) -> None:
        """Verify 'y' parses to True."""
        parser = BooleanParser()

        result = parser(response="y")

        assert result is True

    def test_parses_1_as_true(self) -> None:
        """Verify '1' parses to True."""
        parser = BooleanParser()

        result = parser(response="1")

        assert result is True

    def test_parses_no_as_false(self) -> None:
        """Verify 'no' parses to False."""
        parser = BooleanParser()

        result = parser(response="no")

        assert result is False

    def test_parses_false_as_false(self) -> None:
        """Verify 'false' parses to False."""
        parser = BooleanParser()

        result = parser(response="false")

        assert result is False

    def test_parses_n_as_false(self) -> None:
        """Verify 'n' parses to False."""
        parser = BooleanParser()

        result = parser(response="n")

        assert result is False

    def test_parses_0_as_false(self) -> None:
        """Verify '0' parses to False."""
        parser = BooleanParser()

        result = parser(response="0")

        assert result is False

    def test_case_insensitive(self) -> None:
        """Verify parsing is case-insensitive."""
        parser = BooleanParser()

        assert parser(response="YES") is True
        assert parser(response="True") is True
        assert parser(response="NO") is False
        assert parser(response="False") is False

    def test_strips_whitespace(self) -> None:
        """Verify whitespace is stripped."""
        parser = BooleanParser()

        result = parser(response="  yes  ")

        assert result is True

    def test_raises_for_invalid_input(self) -> None:
        """Verify parser raises ParserError for invalid input."""
        parser = BooleanParser()

        with pytest.raises(ParserError) as exc_info:
            parser(response="maybe")

        assert "Could not parse as boolean" in exc_info.value.reason

    def test_returns_none_for_decline(self) -> None:
        """Verify parser returns None for decline keywords."""
        parser = BooleanParser()

        result = parser(response="UNKNOWN")

        assert result is None
