from __future__ import annotations

import re
from typing import TYPE_CHECKING

from winnow.parser.base import Parser, ParserError

if TYPE_CHECKING:
    from collections.abc import Callable


class FloatParser(Parser[float]):
    """Parses a floating-point number from an LLM response.

    Optionally extracts a unit suffix and applies a conversion function.
    """

    def __init__(
        self,
        *,
        unit_conversion: Callable[[float, str], float] | None = None,
    ) -> None:
        """Initialise the parser.

        Args:
            unit_conversion: Optional function that takes (value, unit_string)
                and returns the converted value. If None, units are ignored.
        """
        self._unit_conversion = unit_conversion

    def parse(self, response: str) -> float:
        """Parse a floating-point number from the response.

        Raises:
            ParserError: If no number can be extracted.
        """
        match = re.search(r"(-?[\d.]+)\s*(\w*)", response.strip())
        if not match:
            raise ParserError(response=response, reason="Could not extract number")

        try:
            value = float(match.group(1))
        except ValueError:
            raise ParserError(response=response, reason="Could not parse as float")

        unit = match.group(2).strip()
        if unit and self._unit_conversion is not None:
            value = self._unit_conversion(value, unit)

        return value
