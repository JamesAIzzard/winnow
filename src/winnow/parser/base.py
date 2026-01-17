from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from winnow.config import default_config
from winnow.exceptions import ModelDeclinedError

T = TypeVar("T")


class Parser(ABC, Generic[T]):
    """Base class for parsers that convert LLM responses into typed values.

    Subclasses must implement the `parse` method. The `__call__` method handles
    decline detection before delegating to `parse`.
    """

    def __call__(self, *, response: str) -> T:
        """Parse a response string.

        Returns the parsed value.

        Raises:
            ModelDeclinedError: If the model declined to answer.
            ParseFailedError: If the response cannot be parsed.
        """
        normalised = response.strip().upper()
        if normalised == default_config.decline_keyword:
            raise ModelDeclinedError(response=response)
        return self.parse(response)

    @abstractmethod
    def parse(self, response: str) -> T:
        """Parse a non-decline response into the target type.

        Raises:
            ParseFailedError: If parsing fails.
        """
        ...
