from __future__ import annotations

import pytest

from winnow.exceptions import ModelDeclinedError, ParseFailedError
from winnow.parser.float_literal_pair import FloatLiteralPairParser


class TestFloatLiteralPairParser:
    def test_parses_valid_pair(self) -> None:
        """Verify parser returns a float-literal tuple for valid input."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"grams", "millilitres", "whole"}),
        )

        result = parser(response="100 grams")

        assert result == (100.0, "grams")

    def test_parses_decimal_float(self) -> None:
        """Verify parser handles decimal float values."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"grams", "whole"}),
        )

        result = parser(response="2.5 whole")

        assert result == (2.5, "whole")

    def test_parses_negative_float(self) -> None:
        """Verify parser handles negative float values."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"grams"}),
        )

        result = parser(response="-10 grams")

        assert result == (-10.0, "grams")

    def test_case_insensitive_by_default(self) -> None:
        """Verify literal matching is case-insensitive by default."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"grams", "millilitres"}),
        )

        result = parser(response="100 GRAMS")

        assert result == (100.0, "grams")

    def test_case_sensitive_when_specified(self) -> None:
        """Verify case-sensitive mode rejects mismatched casing."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"Grams", "Whole"}),
            case_sensitive=True,
        )

        result = parser(response="100 Grams")
        assert result == (100.0, "Grams")

        with pytest.raises(ParseFailedError):
            parser(response="100 grams")

    def test_float_constraint_rejects_invalid_value(self) -> None:
        """Verify float constraint failure raises ParseFailedError."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"grams"}),
            float_constraint=lambda x: x > 0,
        )

        with pytest.raises(ParseFailedError) as exc_info:
            parser(response="-5 grams")

        assert exc_info.value.response == "-5 grams"

    def test_float_constraint_allows_valid_value(self) -> None:
        """Verify float constraint passes for valid values."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"grams"}),
            float_constraint=lambda x: x > 0,
        )

        result = parser(response="100 grams")

        assert result == (100.0, "grams")

    def test_raises_for_invalid_literal(self) -> None:
        """Verify parser raises ParseFailedError for unrecognised literal."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"grams", "whole"}),
        )

        with pytest.raises(ParseFailedError) as exc_info:
            parser(response="100 bananas")

        assert exc_info.value.response == "100 bananas"

    def test_raises_for_missing_float(self) -> None:
        """Verify parser raises ParseFailedError when no number is present."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"grams"}),
        )

        with pytest.raises(ParseFailedError) as exc_info:
            parser(response="grams")

        assert exc_info.value.response == "grams"

    def test_raises_for_missing_literal(self) -> None:
        """Verify parser raises ParseFailedError when no literal is present."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"grams"}),
        )

        with pytest.raises(ParseFailedError) as exc_info:
            parser(response="100")

        assert exc_info.value.response == "100"

    def test_raises_for_decline(self) -> None:
        """Verify parser raises ModelDeclinedError for decline keyword."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"grams"}),
        )

        with pytest.raises(ModelDeclinedError):
            parser(response="DECLINE")

    def test_strips_whitespace_around_literal(self) -> None:
        """Verify parser handles extra whitespace between components."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"grams"}),
        )

        result = parser(response="  100   grams  ")

        assert result == (100.0, "grams")


class TestLiteralAliases:
    def test_alias_resolves_to_canonical_name(self) -> None:
        """Verify an alias in the response resolves to the canonical option."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"gram"}),
            literal_aliases={"gram": ["grams"]},
        )

        result = parser(response="0 grams")

        assert result == (0.0, "gram")

    def test_canonical_name_still_accepted(self) -> None:
        """Verify the canonical name is still matched when aliases are provided."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"gram"}),
            literal_aliases={"gram": ["grams"]},
        )

        result = parser(response="100 gram")

        assert result == (100.0, "gram")

    def test_multiple_aliases_for_one_option(self) -> None:
        """Verify multiple aliases all resolve to the same canonical name."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"gram"}),
            literal_aliases={"gram": ["grams", "g"]},
        )

        assert parser(response="100 grams") == (100.0, "gram")
        assert parser(response="100 g") == (100.0, "gram")

    def test_aliases_are_case_insensitive_by_default(self) -> None:
        """Verify alias matching respects the default case-insensitive mode."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"gram"}),
            literal_aliases={"gram": ["grams"]},
        )

        result = parser(response="50 GRAMS")

        assert result == (50.0, "gram")

    def test_unknown_literal_still_raises(self) -> None:
        """Verify unrecognised literals still raise even when aliases are set."""
        parser = FloatLiteralPairParser(
            literal_options=frozenset({"gram"}),
            literal_aliases={"gram": ["grams"]},
        )

        with pytest.raises(ParseFailedError):
            parser(response="100 bananas")
