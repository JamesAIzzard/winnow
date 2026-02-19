from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping

from winnow.exceptions import ParseFailedError
from winnow.parser.base import Parser


class FloatLiteralPairParser(Parser[tuple[float, str]]):
    """Parses a compound float-literal response such as '100 grams' or '1 whole'.

    The float component is extracted first, then the remainder is matched
    against the set of known literal options (and their aliases, if provided).
    Aliases resolve back to the canonical option name.
    """

    def __init__(
        self,
        literal_options: frozenset[str],
        *,
        literal_aliases: Mapping[str, Collection[str]] | None = None,
        float_constraint: Callable[[float], bool] | None = None,
        case_sensitive: bool = False,
    ) -> None:
        """Initialise the parser.

        Args:
            literal_options: The set of valid string values for the literal component.
            literal_aliases: Optional mapping from canonical option name to
                alternative forms (e.g. ``{"gram": ["grams", "g"]}``).
                Each alias resolves to its canonical name during matching.
            float_constraint: Optional validation function applied to the parsed float.
                If provided and returns False, ParseFailedError is raised.
            case_sensitive: Whether literal matching is case-sensitive.
        """
        self._float_constraint = float_constraint
        self._case_sensitive = case_sensitive
        self._lookup: dict[str, str] = {}

        for option in literal_options:
            key = option if case_sensitive else option.lower()
            self._lookup[key] = option

        if literal_aliases is not None:
            for canonical, aliases in literal_aliases.items():
                for alias in aliases:
                    key = alias if case_sensitive else alias.lower()
                    self._lookup[key] = canonical

    def parse(self, response: str) -> tuple[float, str]:
        """Parse a float-literal pair from the response.

        Raises:
            ParseFailedError: If the float cannot be extracted, the literal
                does not match any option, or the float constraint fails.
        """
        value = self._extract_float(response)
        if self._float_constraint is not None and not self._float_constraint(value):
            raise ParseFailedError(response=response)

        literal = self._match_literal(response)
        return (value, literal)

    def _extract_float(self, response: str) -> float:
        """Extract the first floating-point number from the response.

        Raises:
            ParseFailedError: If no number can be found.
        """
        match = re.search(r"-?[\d.]+", response)
        if not match:
            raise ParseFailedError(response=response)

        try:
            return float(match.group(0))
        except ValueError:
            raise ParseFailedError(response=response)

    def _match_literal(self, response: str) -> str:
        """Match the non-numeric remainder against the known options.

        Raises:
            ParseFailedError: If no option matches.
        """
        remainder = re.sub(r"-?[\d.]+", "", response).strip()
        key = remainder if self._case_sensitive else remainder.lower()

        if key not in self._lookup:
            raise ParseFailedError(response=response)
        return self._lookup[key]
