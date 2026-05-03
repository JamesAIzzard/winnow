"""Winnow: Statistically robust data extraction from large language models."""

from __future__ import annotations


# Core types
from winnow.results import Estimate, NeedsReview
from winnow.state import NoEstimate, ReviewReason, SampleState, SampleStatus

# Question system
from winnow.question import Question, QuestionBank

# LLM client
from winnow.llm_client import LLMClient

# Main entry point
from winnow.collect import collect

# Parser package
from winnow.parser import (
    BooleanParser,
    FloatLiteralPairParser,
    FloatParser,
    LiteralParser,
    OptionalBoundedIntParser,
    Parser,
)

# Estimator package
from winnow.estimator import (
    BooleanEstimator,
    CategoricalEstimator,
    Estimator,
    NumericalEstimator,
    OpenCategoricalEstimator,
    OptionalIntEstimator,
)

# Stopping criterion
from winnow.stopping import StoppingCriterion

# Logging
from winnow.jsonl_logging import configure_jsonl_logging

# Exceptions
from winnow.exceptions import (
    EstimationFailedError,
    ModelDeclinedError,
    ParseFailedError,
    UnknownInitialStateError,
    WinnowError,
)


__all__ = [
    # LLM client
    "LLMClient",
    # Core types
    "Estimate",
    "NeedsReview",
    "NoEstimate",
    "ReviewReason",
    "SampleState",
    "SampleStatus",
    # Question system
    "Question",
    "QuestionBank",
    # Main entry point
    "collect",
    # Parser package
    "BooleanParser",
    "FloatLiteralPairParser",
    "FloatParser",
    "LiteralParser",
    "OptionalBoundedIntParser",
    "Parser",
    # Estimator package
    "BooleanEstimator",
    "CategoricalEstimator",
    "Estimator",
    "NumericalEstimator",
    "OpenCategoricalEstimator",
    "OptionalIntEstimator",
    # Stopping criterion
    "StoppingCriterion",
    # Logging
    "configure_jsonl_logging",
    # Exceptions
    "EstimationFailedError",
    "ModelDeclinedError",
    "ParseFailedError",
    "UnknownInitialStateError",
    "WinnowError",
]
