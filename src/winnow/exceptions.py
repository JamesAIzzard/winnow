"""Winnow exceptions."""

from __future__ import annotations


class WinnowError(Exception):
    """Base exception for all Winnow errors."""

    ...


class ParseFailedError(WinnowError):
    """Raised when a parser fails to parse a response."""

    def __init__(self, *, response: str, reason: str) -> None:
        self.response = response
        self.reason = reason
        super().__init__(f"{reason}: {response!r}")
