"""Winnow exceptions."""

from __future__ import annotations


class WinnowError(Exception):
    """Base exception for all Winnow errors."""

    ...


class ParseFailedError(WinnowError):
    """Raised when a parser fails to parse a response."""

    def __init__(self, *, response: str) -> None:
        self.response = response
        super().__init__(f"Could not parse response: {response!r}")


class EstimationFailedError(WinnowError):
    """Raised when estimation fails due to insufficient valid samples."""

    def __init__(self, *, question_uid: str) -> None:
        self.question_uid = question_uid
        super().__init__(f"Failed to estimate '{question_uid}': no valid samples collected")


class ModelDeclinedError(WinnowError):
    """Raised when the model explicitly declines to answer."""

    def __init__(self, *, response: str) -> None:
        self.response = response
        super().__init__(f"Model declined to answer: {response!r}")
