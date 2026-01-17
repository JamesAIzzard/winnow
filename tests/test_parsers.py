from __future__ import annotations

import pytest

from winnow.parsers import (
    parse_boolean,
    parse_float,
    parse_integer,
    StringChoiceParser,
    StringListParser,
)
from winnow.exceptions import ParseFailedError


class TestParseBoolean:
    def test_true_and_yes(self) -> None:
        """Verify returns True for 'true' and 'yes' (case-insensitive)."""
        assert parse_boolean(response="true") is True
        assert parse_boolean(response="yes") is True
        assert parse_boolean(response="TrUe") is True
        assert parse_boolean(response="YeS") is True

    def test_false_and_no(self) -> None:
        """Verify returns False for 'false' and 'no' (case-insensitive)."""
        assert parse_boolean(response="false") is False
        assert parse_boolean(response="no") is False
        assert parse_boolean(response="FaLsE") is False
        assert parse_boolean(response="NO") is False

    def test_whitespace_is_ignored(self) -> None:
        """Verify strips leading and trailing whitespace before parsing."""
        assert parse_boolean(response="  yes \n") is True
        assert parse_boolean(response="\t no \t") is False

    def test_invalid_raises(self) -> None:
        """Verify raises ParseFailedError when input is unrecognised."""
        with pytest.raises(ParseFailedError):
            parse_boolean(response="maybe")


class TestParseFloat:
    def test_basic_and_scientific(self) -> None:
        """Verify parses standard and scientific-notation floats."""
        assert parse_float(response="3.14") == pytest.approx(3.14)
        assert parse_float(response="1e3") == pytest.approx(1000.0)
        assert parse_float(response="-2.5") == pytest.approx(-2.5)

    def test_whitespace_is_ignored(self) -> None:
        """Verify strips whitespace before float conversion."""
        assert parse_float(response="  42.0 \n") == pytest.approx(42.0)

    def test_invalid_raises(self) -> None:
        """Verify raises ParseFailedError when value is not a float."""
        with pytest.raises(ParseFailedError):
            parse_float(response="abc")
        with pytest.raises(ParseFailedError):
            parse_float(response="1.2.3")


class TestParseInteger:
    def test_basic_and_negative(self) -> None:
        """Verify parses basic and negative integers."""
        assert parse_integer(response="7") == 7
        assert parse_integer(response="-12") == -12

    def test_whitespace_is_ignored(self) -> None:
        """Verify strips whitespace before integer conversion."""
        assert parse_integer(response="  15 \t") == 15

    def test_invalid_raises(self) -> None:
        """Verify raises ParseFailedError when value is not an integer."""
        with pytest.raises(ParseFailedError):
            parse_integer(response="3.0")
        with pytest.raises(ParseFailedError):
            parse_integer(response="ten")


class TestStringChoiceParser:
    def test_basic_comma_separated_valid_choices(self) -> None:
        """Verify returns only valid choices parsed via list logic."""
        parser = StringChoiceParser({"red", "green", "blue"})
        assert parser(response="red,blue") == ["red", "blue"]

    def test_strips_whitespace_and_ignores_empty_segments(self) -> None:
        """Verify trims items and discards blanks before validation."""
        parser = StringChoiceParser({"alpha", "beta", "gamma"})
        assert parser(response="  alpha , , beta ,  gamma  ") == [
            "alpha",
            "beta",
            "gamma",
        ]

    def test_case_sensitivity(self) -> None:
        """Verify membership check is case-sensitive and fails on mismatch."""
        parser = StringChoiceParser({"yes", "no"})
        with pytest.raises(ParseFailedError):
            parser(response="Yes")

    def test_any_invalid_raises(self) -> None:
        """Verify raises when any parsed item is not in the allowed choices."""
        parser = StringChoiceParser({"cat", "dog"})
        with pytest.raises(ParseFailedError):
            parser(response="cat,hamster")

    def test_custom_separator_and_allow_empty(self) -> None:
        """Verify supports custom separator and allow_empty passthrough."""
        parser = StringChoiceParser({"a", "b"}, separator=";", allow_empty=True)
        assert parser(response=" ; ; ") == []
        assert parser(response="a; b") == ["a", "b"]

    def test_case_insensitive_option_accepts_mixed_case(self) -> None:
        """Verify honours case_sensitive=False and returns canonical choices."""
        parser = StringChoiceParser({"yes", "no"}, case_sensitive=False)
        assert parser(response="Yes,NO") == ["yes", "no"]

    def test_case_insensitive_collapses_duplicate_choices(self) -> None:
        """Verify duplicates in choices collapse to a single canonical option."""
        parser = StringChoiceParser({"YES", "yes", "No"}, case_sensitive=False)
        assert parser(response="YES,yes,No,no") == ["yes", "yes", "No", "No"]


class TestStringListParser:
    def test_basic_comma_separated(self) -> None:
        """Verify splits a simple comma-separated list into trimmed items."""
        parser = StringListParser()
        assert parser(response="a,b,c") == ["a", "b", "c"]

    def test_strips_whitespace_and_ignores_empty(self) -> None:
        """Verify trims items and discards empty segments and blanks."""
        parser = StringListParser()
        assert parser(response="  one , two,,  , three  ") == [
            "one",
            "two",
            "three",
        ]

    def test_custom_separator(self) -> None:
        """Verify supports a custom separator character."""
        parser = StringListParser(separator=";")
        assert parser(response="alpha; beta;gamma") == ["alpha", "beta", "gamma"]

    def test_all_empty_raises(self) -> None:
        """Verify raises ParseFailedError when no non-empty items are present."""
        parser = StringListParser()
        with pytest.raises(ParseFailedError):
            parser(response="   ")
        with pytest.raises(ParseFailedError):
            parser(response=", , ,")

    def test_allow_empty_returns_empty_list(self) -> None:
        """Verify returns empty list when allow_empty=True and no items parsed."""
        parser = StringListParser(allow_empty=True)
        assert parser(response="   ") == []
        assert parser(response=", , ,") == []
        assert parser(response="") == []

    def test_allow_empty_with_custom_separator(self) -> None:
        """Verify allow_empty works with a custom separator too."""
        parser = StringListParser(separator=";", allow_empty=True)
        assert parser(response=" ; ; ") == []
