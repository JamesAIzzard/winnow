from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from winnow.state import NoEstimate, SampleState, SampleStatus

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture
def make_state():
    """Create a SampleState from a sequence of samples."""

    def _make_state(samples: Sequence) -> SampleState:
        return SampleState(
            samples=tuple(samples),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
            current_estimate=NoEstimate,
            current_confidence=0.0,
            status=SampleStatus.COLLECTING,
            failure_reason=None,
        )

    return _make_state
