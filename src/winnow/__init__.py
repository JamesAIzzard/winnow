"""Winnow: Statistically robust data extraction from large language models."""

from __future__ import annotations

# Core types
from winnow.types import Archetype as Archetype
from winnow.types import Estimate as Estimate
from winnow.types import SampleState as SampleState

# Question system
from winnow.question import Question as Question
from winnow.question import QuestionBank as QuestionBank

# Main entry point
from winnow.collect import collect as collect

# Parser package
from winnow.parser import BooleanParser as BooleanParser
from winnow.parser import FloatParser as FloatParser
from winnow.parser import LiteralParser as LiteralParser
from winnow.parser import Parser as Parser

# Estimator package
from winnow.estimator import BooleanEstimator as BooleanEstimator
from winnow.estimator import CategoricalEstimator as CategoricalEstimator
from winnow.estimator import ConsensusEstimator as ConsensusEstimator
from winnow.estimator import NumericalEstimator as NumericalEstimator

# Stopping criteria
from winnow.stopping import All as All
from winnow.stopping import Any as Any
from winnow.stopping import ConfidenceReached as ConfidenceReached
from winnow.stopping import ConsecutiveDeclines as ConsecutiveDeclines
from winnow.stopping import MaxQueries as MaxQueries
from winnow.stopping import MinSamples as MinSamples
from winnow.stopping import StoppingCriterion as StoppingCriterion
from winnow.stopping import UnanimousAgreement as UnanimousAgreement
from winnow.stopping import categorical_stopping as categorical_stopping
from winnow.stopping import relaxed_stopping as relaxed_stopping
from winnow.stopping import standard_stopping as standard_stopping
