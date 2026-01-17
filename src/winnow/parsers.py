from __future__ import annotations

from .types import Parser
from .exceptions import ParseFailedError


def parse_string(*, response: str) -> str:
    if not response.strip():
        raise ParseFailedError(response=response, reason="Empty string")

    return response


def parse_boolean(*, response: str) -> bool:
    normalized = response.strip().lower()
    if normalized in {"true", "yes"}:
        return True
    if normalized in {"false", "no"}:
        return False
    raise ParseFailedError(response=response, reason="Not a boolean")


def parse_float(*, response: str) -> float:
    try:
        return float(response.strip())
    except ValueError:
        raise ParseFailedError(response=response, reason="Not a float")


def parse_integer(*, response: str) -> int:
    try:
        return int(response.strip())
    except ValueError:
        raise ParseFailedError(response=response, reason="Not an integer")


class StringListParser(Parser[list[str]]):
    def __init__(self, *, separator: str = ",", allow_empty: bool = False):
        self._separator = separator
        self._allow_empty = allow_empty

    def __call__(self, *, response: str) -> list[str]:
        items = [
            item.strip() for item in response.split(self._separator) if item.strip()
        ]
        if not items and not self._allow_empty:
            raise ParseFailedError(response=response, reason="Empty list")
        return items


class StringChoiceParser(Parser[list[str]]):
    def __init__(
        self,
        choices: set[str],
        *,
        separator: str = ",",
        allow_empty: bool = False,
        case_sensitive: bool = True,
    ):
        self._choices = choices
        self._case_sensitive = case_sensitive
        self._list_parser = StringListParser(
            separator=separator, allow_empty=allow_empty
        )
        self._lower_map: dict[str, str] | None

        if not self._case_sensitive:
            groups: dict[str, set[str]] = {}
            for choice in self._choices:
                key = choice.lower()
                groups.setdefault(key, set()).add(choice)

            def choose_canonical(variants: set[str]) -> str:
                # Prefer the fully lower-case variant if present,
                # else lexicographically first.
                return sorted(variants, key=lambda s: (s != s.lower(), s))[0]

            self._lower_map = {
                key: choose_canonical(variants) for key, variants in groups.items()
            }
        else:
            self._lower_map = None

    def __call__(self, *, response: str) -> list[str]:
        items = self._list_parser(response=response)
        if self._case_sensitive:
            if any(item not in self._choices for item in items):
                raise ParseFailedError(response=response, reason="Invalid choice")
            return items

        assert self._lower_map is not None
        result: list[str] = []
        for item in items:
            key = item.lower()
            if key not in self._lower_map:
                raise ParseFailedError(response=response, reason="Invalid choice")
            result.append(self._lower_map[key])
        return result
