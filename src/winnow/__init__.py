"""Winnow: Statistically robust data extraction from large language models."""

from __future__ import annotations

# Main entry point
from .collect import collect

# Estimator package
from .estimator import (
    BooleanEstimator,
    CategoricalEstimator,
    Estimator,
    NumericalEstimator,
    OpenCategoricalEstimator,
    OptionalIntEstimator,
)

# Exceptions
from .exceptions import (
    EstimationFailedError,
    ModelDeclinedError,
    ParseFailedError,
    UnknownInitialStateError,
    WinnowError,
)

# Exchange boundary
from .exchange.client import LLMClient
from .exchange.logging import configure_jsonl_logging

# Parser package
from .parser import (
    BooleanParser,
    FloatLiteralPairParser,
    FloatParser,
    LiteralParser,
    OptionalBoundedIntParser,
    Parser,
)

# Question system
from .question import Prompt, Question, QuestionBank

# Core types
from .results import Estimate, NeedsReview
from .state import NoEstimate, ReviewReason, SampleState, SampleStatus

# Stopping criterion
from .stopping import StoppingCriterion

__all__ = [
    "BooleanEstimator",
    "BooleanParser",
    "CategoricalEstimator",
    "Estimate",
    "EstimationFailedError",
    "Estimator",
    "FloatLiteralPairParser",
    "FloatParser",
    "LLMClient",
    "LiteralParser",
    "ModelDeclinedError",
    "NeedsReview",
    "NoEstimate",
    "NumericalEstimator",
    "OpenCategoricalEstimator",
    "OptionalBoundedIntParser",
    "OptionalIntEstimator",
    "ParseFailedError",
    "Parser",
    "Prompt",
    "Question",
    "QuestionBank",
    "ReviewReason",
    "SampleState",
    "SampleStatus",
    "StoppingCriterion",
    "UnknownInitialStateError",
    "WinnowError",
    "collect",
    "configure_jsonl_logging",
]
