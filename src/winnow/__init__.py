"""Winnow: Statistically robust data extraction from large language models."""

from __future__ import annotations


# Core types
from winnow.types import Estimate, NeedsReview, NoEstimate, ReviewReason, SampleState, SampleStatus

# Question system
from winnow.question import Question, QuestionBank

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

# Exceptions
from winnow.exceptions import (
    EstimationFailedError,
    ModelDeclinedError,
    ParseFailedError,
    UnknownInitialStateError,
    WinnowError,
)


__all__ = [
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
    # Exceptions
    "EstimationFailedError",
    "ModelDeclinedError",
    "ParseFailedError",
    "UnknownInitialStateError",
    "WinnowError",
]
