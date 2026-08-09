from __future__ import annotations

from winnow import (
    NoEstimate,
    ReviewReason,
    SampleState,
    SampleStatus,
    StoppingCriterion,
)


class TestSampleState:
    def test_query_count_includes_all_attempts(self) -> None:
        """Verify query_count sums samples, declines, and parse failures."""
        state: SampleState[float] = SampleState(
            samples=(1.0, 2.0, 3.0),
            decline_count=2,
            parse_failure_count=1,
            consecutive_declines=0,
            current_estimate=NoEstimate,
            current_confidence=0.0,
            status=SampleStatus.COLLECTING,
            failure_reason=None,
        )

        assert state.query_count == 6

    def test_record_sample_updates_successful_sample_fields(self) -> None:
        """Verify record_sample returns the next successful collection state."""
        state: SampleState[float] = SampleState(
            samples=(1.0,),
            decline_count=1,
            parse_failure_count=2,
            consecutive_declines=1,
            current_estimate=NoEstimate,
            current_confidence=0.0,
            status=SampleStatus.NEEDS_REVIEW,
            failure_reason=ReviewReason.MAX_QUERIES,
        )

        updated = state.record_sample(
            2.0,
            estimate=1.5,
            confidence=0.9,
            effective_sample_count=2,
        )

        assert updated.samples == (1.0, 2.0)
        assert updated.decline_count == 1
        assert updated.parse_failure_count == 2
        assert updated.consecutive_declines == 0
        assert updated.current_estimate == 1.5
        assert updated.current_confidence == 0.9
        assert updated.status is SampleStatus.COLLECTING
        assert updated.failure_reason is None
        assert updated.effective_sample_count == 2

    def test_record_decline_updates_decline_fields(self) -> None:
        """Verify record_decline returns the next declined collection state."""
        state: SampleState[float] = SampleState(
            samples=(1.0,),
            decline_count=1,
            parse_failure_count=2,
            consecutive_declines=3,
            current_estimate=1.0,
            current_confidence=0.8,
            status=SampleStatus.NEEDS_REVIEW,
            failure_reason=ReviewReason.INSUFFICIENT_CONFIDENCE,
            effective_sample_count=1,
        )

        updated = state.record_decline()

        assert updated.samples == (1.0,)
        assert updated.decline_count == 2
        assert updated.parse_failure_count == 2
        assert updated.consecutive_declines == 4
        assert updated.current_estimate == 1.0
        assert updated.current_confidence == 0.8
        assert updated.status is SampleStatus.COLLECTING
        assert updated.failure_reason is None
        assert updated.effective_sample_count == 1

    def test_record_parse_failure_updates_parse_failure_fields(self) -> None:
        """Verify record_parse_failure returns the next failed parse state."""
        state: SampleState[float] = SampleState(
            samples=(1.0,),
            decline_count=1,
            parse_failure_count=2,
            consecutive_declines=3,
            current_estimate=1.0,
            current_confidence=0.8,
            status=SampleStatus.NEEDS_REVIEW,
            failure_reason=ReviewReason.INSUFFICIENT_CONFIDENCE,
            effective_sample_count=1,
        )

        updated = state.record_parse_failure()

        assert updated.samples == (1.0,)
        assert updated.decline_count == 1
        assert updated.parse_failure_count == 3
        assert updated.consecutive_declines == 0
        assert updated.current_estimate == 1.0
        assert updated.current_confidence == 0.8
        assert updated.status is SampleStatus.COLLECTING
        assert updated.failure_reason is None
        assert updated.effective_sample_count == 1

    def test_resolve_status_returns_same_state_when_collection_continues(self) -> None:
        """Verify resolve_status leaves a non-terminal state unchanged."""
        state: SampleState[float] = SampleState(
            samples=(1.0,),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
            current_estimate=1.0,
            current_confidence=0.8,
            status=SampleStatus.COLLECTING,
            failure_reason=None,
        )

        updated = state.resolve_status(
            StoppingCriterion(min_samples=2, confidence_threshold=0.9),
        )

        assert updated is state

    def test_resolve_status_marks_converged_when_thresholds_are_met(self) -> None:
        """Verify resolve_status marks terminal success when thresholds are met."""
        state: SampleState[float] = SampleState(
            samples=(1.0, 1.1),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
            current_estimate=1.05,
            current_confidence=0.95,
            status=SampleStatus.COLLECTING,
            failure_reason=None,
        )

        updated = state.resolve_status(
            StoppingCriterion(min_samples=2, confidence_threshold=0.9),
        )

        assert updated.status is SampleStatus.CONVERGED
        assert updated.failure_reason is None

    def test_resolve_status_marks_review_with_failure_reason(self) -> None:
        """Verify resolve_status marks terminal failure with the review reason."""
        state: SampleState[float] = SampleState(
            samples=(1.0,),
            decline_count=0,
            parse_failure_count=3,
            consecutive_declines=0,
            current_estimate=1.0,
            current_confidence=0.8,
            status=SampleStatus.COLLECTING,
            failure_reason=None,
        )

        updated = state.resolve_status(
            StoppingCriterion(max_parse_failures=3),
        )

        assert updated.status is SampleStatus.NEEDS_REVIEW
        assert updated.failure_reason is ReviewReason.MAX_PARSE_FAILURES
