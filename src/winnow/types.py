from __future__ import annotations
from typing import Protocol, TypeVar, Optional

T = TypeVar("T", covariant=True)


class Parser(Protocol[T]):
    def __call__(self, *, response: str) -> T: ...


class ParsedOpenAIClient(Protocol):
    def ensure(
        self,
        *,
        prompt: str,
        parser: Parser[T],
        max_retries: Optional[int] = None,
        allow_decline: Optional[bool] = None,
    ) -> T: ...

class ParsedOpenAIAPI(Protocol):
    def create_client(
        self,
        model: Optional[str] = None,
        std_preface: Optional[str] = None,
        max_retries: Optional[int] = None,
        allow_decline: Optional[bool] = None,
    ) -> ParsedOpenAIClient: ...