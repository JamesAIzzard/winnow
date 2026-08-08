"""Winnow: Statistically robust data extraction from large language models."""

from __future__ import annotations

# Main entry point
from winnow.collect import collect

# Estimator package
from winnow.estimator import (
    BooleanEstimator,
    CategoricalEstimator,
    Estimator,
    NumericalEstimator,
    OpenCategoricalEstimator,
    OptionalIntEstimator,
)

# Exceptions
from winnow.exceptions import (
    EstimationFailedError,
    ModelDeclinedError,
    ParseFailedError,
    UnknownInitialStateError,
    WinnowError,
)

# Exchange boundary
from winnow.exchange.client import LLMClient
from winnow.exchange.logging import configure_jsonl_logging

# Parser package
from winnow.parser import (
    BooleanParser,
    FloatLiteralPairParser,
    FloatParser,
    LiteralParser,
    OptionalBoundedIntParser,
    Parser,
)

# Question system
from winnow.question import Prompt, Question, QuestionBank

# Core types
from winnow.results import Estimate, NeedsReview
from winnow.state import NoEstimate, ReviewReason, SampleState, SampleStatus

# Stopping criterion
from winnow.stopping import StoppingCriterion

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
