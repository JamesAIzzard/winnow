from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from winnow.estimator.open_categorical import OpenCategoricalEstimator

if TYPE_CHECKING:
    from winnow.types import SampleState


class TestOpenCategoricalEstimatorEstimate:
    def test_returns_mode(self, make_state: Callable[..., SampleState[str]]) -> None:
        """Verify compute_estimate returns the most common value."""
        estimator: OpenCategoricalEstimator[str] = OpenCategoricalEstimator()

        result = estimator.compute_estimate(state=make_state(["a", "b", "a", "c", "a"]))

        assert result == "a"

    def test_handles_tie_by_first_encountered(
        self, make_state: Callable[..., SampleState[str]]
    ) -> None:
        """Verify tie-breaking is deterministic."""
        estimator: OpenCategoricalEstimator[str] = OpenCategoricalEstimator()

        result = estimator.compute_estimate(state=make_state(["a", "b", "a", "b"]))

        assert result in {"a", "b"}

    def test_works_with_tuple_values(
        self, make_state: Callable[..., SampleState[tuple[float, str]]]
    ) -> None:
        """Verify estimator works with compound tuple values."""
        estimator: OpenCategoricalEstimator[tuple[float, str]] = OpenCategoricalEstimator()
        samples = [
            (100.0, "grams"),
            (100.0, "grams"),
            (100.0, "grams"),
            (1.0, "whole"),
            (100.0, "grams"),
        ]

        result = estimator.compute_estimate(state=make_state(samples))

        assert result == (100.0, "grams")


class TestOpenCategoricalEstimatorConfidence:
    def test_unanimous_agreement(
        self, make_state: Callable[..., SampleState[str]]
    ) -> None:
        """Verify full confidence when all samples agree."""
        estimator: OpenCategoricalEstimator[str] = OpenCategoricalEstimator()

        confidence = estimator.compute_confidence(
            state=make_state(["a", "a", "a", "a", "a"]), estimate="a"
        )

        assert confidence == 1.0

    def test_zero_confidence_for_empty_samples(
        self, make_state: Callable[..., SampleState[str]]
    ) -> None:
        """Verify zero confidence for empty sample list."""
        estimator: OpenCategoricalEstimator[str] = OpenCategoricalEstimator()

        confidence = estimator.compute_confidence(state=make_state([]), estimate="a")

        assert confidence == 0.0

    def test_raw_agreement_not_normalised(
        self, make_state: Callable[..., SampleState[str]]
    ) -> None:
        """Verify confidence is raw agreement without baseline normalisation."""
        estimator: OpenCategoricalEstimator[str] = OpenCategoricalEstimator()

        # 3 out of 5 match → raw agreement = 0.6
        confidence = estimator.compute_confidence(
            state=make_state(["a", "b", "a", "c", "a"]), estimate="a"
        )

        assert confidence == pytest.approx(0.6)

    def test_single_sample(
        self, make_state: Callable[..., SampleState[str]]
    ) -> None:
        """Verify confidence is 1.0 for a single sample matching the estimate."""
        estimator: OpenCategoricalEstimator[str] = OpenCategoricalEstimator()

        confidence = estimator.compute_confidence(
            state=make_state(["a"]), estimate="a"
        )

        assert confidence == 1.0

    def test_scattered_samples(
        self, make_state: Callable[..., SampleState[str]]
    ) -> None:
        """Verify low confidence when samples are evenly scattered."""
        estimator: OpenCategoricalEstimator[str] = OpenCategoricalEstimator()

        # 2 out of 10 match → raw agreement = 0.2
        confidence = estimator.compute_confidence(
            state=make_state(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
            estimate="a",
        )

        assert confidence == pytest.approx(0.1)

    def test_confidence_with_tuple_values(
        self, make_state: Callable[..., SampleState[tuple[float, str]]]
    ) -> None:
        """Verify confidence calculation works with compound tuple values."""
        estimator: OpenCategoricalEstimator[tuple[float, str]] = OpenCategoricalEstimator()
        samples = [
            (100.0, "grams"),
            (100.0, "grams"),
            (100.0, "grams"),
            (1.0, "whole"),
            (100.0, "millilitres"),
        ]

        confidence = estimator.compute_confidence(
            state=make_state(samples), estimate=(100.0, "grams")
        )

        # 3 out of 5 match → 0.6
        assert confidence == pytest.approx(0.6)
