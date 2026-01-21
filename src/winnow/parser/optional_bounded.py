from __future__ import annotations

from winnow.exceptions import ParseFailedError
from winnow.parser.base import Parser


class OptionalBoundedIntParser(Parser[int | None]):
    """Parses an integer within bounds, or None if not applicable."""

    def __init__(self, *, min_value: int, max_value: int) -> None:
        self._min_value = min_value
        self._max_value = max_value

    def parse(self, response: str) -> int | None:
        """Parse the response as a bounded integer or None."""
        normalised = response.strip().lower()

        if normalised == "none":
            return None

        try:
            value = int(normalised)
        except ValueError:
            raise ParseFailedError(response=response)

        if value < self._min_value or value > self._max_value:
            raise ParseFailedError(response=response)

        return value
