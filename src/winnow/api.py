from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from .client import _ParsedOpenAIClient
from .types import ParsedOpenAIAPI

if TYPE_CHECKING:
    from .types import ParsedOpenAIClient


class _ParsedOpenAIAPI(ParsedOpenAIAPI):
    def __init__(
        self,
        *,
        default_model: str,
        default_std_preface: Optional[str] = None,
        default_max_retries: int,
        default_allow_decline: bool,
    ):
        self._default_model = default_model
        self._default_std_preface = default_std_preface
        self._default_max_retries = default_max_retries
        self._default_allow_decline = default_allow_decline

    def create_client(
        self,
        model: Optional[str] = None,
        std_preface: Optional[str] = None,
        max_retries: Optional[int] = None,
        allow_decline: Optional[bool] = None,
    ) -> ParsedOpenAIClient:
        return _ParsedOpenAIClient(
            model=model or self._default_model,
            std_preface=(
                std_preface
                if std_preface is not None
                else self._default_std_preface
            ),
            max_retries=(
                max_retries
                if max_retries is not None
                else self._default_max_retries
            ),
            allow_decline=allow_decline
            if allow_decline is not None
            else self._default_allow_decline,
        )


def create_parsed_openai_api(
    *,
    default_model: str,
    default_std_preface: Optional[str] = None,
    default_max_retries: int = 10,
    default_allow_decline: bool = True,
) -> ParsedOpenAIAPI:
    return _ParsedOpenAIAPI(
        default_model=default_model,
        default_std_preface=default_std_preface,
        default_max_retries=default_max_retries,
        default_allow_decline=default_allow_decline,
    )
