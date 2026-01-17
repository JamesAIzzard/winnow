from __future__ import annotations

import pytest

from winnow.exceptions import ParseFailedError
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

    def test_parses_number_with_unit(self) -> None:
        """Verify parser extracts number before unit."""
        parser = FloatParser()

        result = parser(response="100g")

        assert result == 100.0

    def test_parses_number_with_space_before_unit(self) -> None:
        """Verify parser handles space between number and unit."""
        parser = FloatParser()

        result = parser(response="50 kg")

        assert result == 50.0

    def test_raises_for_non_numeric(self) -> None:
        """Verify parser raises ParseFailedError for non-numeric input."""
        parser = FloatParser()

        with pytest.raises(ParseFailedError) as exc_info:
            parser(response="hello")

        assert "Could not extract number" in exc_info.value.reason

    def test_returns_none_for_decline(self) -> None:
        """Verify parser returns None for decline keywords."""
        parser = FloatParser()

        result = parser(response="DECLINE")

        assert result is None


class TestFloatParserWithUnitConversion:
    def test_applies_unit_conversion(self) -> None:
        """Verify unit conversion function is applied."""

        def convert(value: float, unit: str) -> float:
            if unit.lower() == "kg":
                return value * 1000
            return value

        parser = FloatParser(unit_conversion=convert)

        result = parser(response="2.5 kg")

        assert result == 2500.0

    def test_no_conversion_without_unit(self) -> None:
        """Verify no conversion when unit is not present."""

        def convert(value: float, unit: str) -> float:
            if unit.lower() == "kg":
                return value * 1000
            return value

        parser = FloatParser(unit_conversion=convert)

        result = parser(response="100")

        assert result == 100.0
