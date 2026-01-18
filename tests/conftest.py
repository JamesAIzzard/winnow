from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from winnow.types import SampleState

if TYPE_CHECKING:
    from collections.abc import Sequence

# Ensure the src/ directory is on sys.path so tests can import the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def make_state():
    """Create a SampleState from a sequence of samples."""

    def _make_state(samples: Sequence) -> SampleState:
        return SampleState(
            samples=tuple(samples),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
        )

    return _make_state
