from __future__ import annotations

from typing import Generic, TypeVar

from winnow.exceptions import ParseFailedError
from winnow.parser.base import Parser

T = TypeVar("T")


class LiteralParser(Parser[T], Generic[T]):
    """Parses a response matching one of a known set of values."""

    def __init__(
        self,
        options: frozenset[T],
        *,
        case_sensitive: bool = False,
    ) -> None:
        """Initialise the parser.

        Args:
            options: The set of valid values.
            case_sensitive: Whether string matching should be case-sensitive.
                Only applies when options are strings.
        """
        self._options = options
        self._case_sensitive = case_sensitive
        self._lookup: dict[str, T] = {}

        for option in options:
            key = str(option) if case_sensitive else str(option).lower()
            self._lookup[key] = option

    def parse(self, response: str) -> T:
        """Parse the response as one of the known options.

        Raises:
            ParseFailedError: If the response does not match any valid option.
        """
        key = response.strip() if self._case_sensitive else response.strip().lower()
        if key not in self._lookup:
            raise ParseFailedError(
                response=response,
                reason=f"Not a valid option. Expected one of: {self._options}",
            )
        return self._lookup[key]
