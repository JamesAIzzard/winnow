from __future__ import annotations

from winnow.exceptions import ParseFailedError
from winnow.parser.base import Parser


class BooleanParser(Parser[bool]):
    """Parses boolean responses such as yes/no, true/false."""

    truthy: frozenset[str] = frozenset({"yes", "true", "1", "y"})
    falsy: frozenset[str] = frozenset({"no", "false", "0", "n"})

    def parse(self, response: str) -> bool:
        """Parse a boolean from the response.

        Raises:
            ParseFailedError: If the response is not recognisable as a boolean.
        """
        normalised = response.strip().lower()
        if normalised in self.truthy:
            return True
        if normalised in self.falsy:
            return False
        raise ParseFailedError(response=response, reason="Could not parse as boolean")
