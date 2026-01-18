from __future__ import annotations

import re

from winnow.exceptions import ParseFailedError
from winnow.parser.base import Parser


class FloatParser(Parser[float]):
    """Parses a floating-point number from an LLM response."""

    def parse(self, response: str) -> float:
        """Parse a floating-point number from the response.

        Raises:
            ParseFailedError: If no number can be extracted.
        """
        match = re.search(r"-?[\d.]+", response)
        if not match:
            raise ParseFailedError(response=response)

        try:
            return float(match.group(0))
        except ValueError:
            raise ParseFailedError(response=response)
