from __future__ import annotations

from winnow.stopping.base import StoppingCriterion as StoppingCriterion
from winnow.stopping.combinators import All as All
from winnow.stopping.combinators import Any as Any
from winnow.stopping.factories import categorical_stopping as categorical_stopping
from winnow.stopping.factories import relaxed_stopping as relaxed_stopping
from winnow.stopping.factories import standard_stopping as standard_stopping
from winnow.stopping.primitives import ConfidenceReached as ConfidenceReached
from winnow.stopping.primitives import ConsecutiveDeclines as ConsecutiveDeclines
from winnow.stopping.primitives import MaxQueries as MaxQueries
from winnow.stopping.primitives import MinSamples as MinSamples
from winnow.stopping.primitives import UnanimousAgreement as UnanimousAgreement
