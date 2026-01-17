from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from winnow.config import default_config

T = TypeVar("T")


class Parser(ABC, Generic[T]):
    """Base class for parsers that convert LLM responses into typed values.

    Subclasses must implement the `parse` method. The `__call__` method handles
    decline detection before delegating to `parse`.
    """

    decline_keywords: frozenset[str] = default_config.decline_keywords

    def __call__(self, *, response: str) -> T | None:
        """Parse a response string.

        Returns the parsed value, or None if the model declined to answer.

        Raises:
            ParseFailedError: If the response cannot be parsed.
        """
        normalised = response.strip().upper()
        if any(keyword in normalised for keyword in self.decline_keywords):
            return None
        return self.parse(response)

    @abstractmethod
    def parse(self, response: str) -> T:
        """Parse a non-decline response into the target type.

        Raises:
            ParseFailedError: If parsing fails.
        """
        ...
