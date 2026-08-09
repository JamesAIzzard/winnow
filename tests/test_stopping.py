from __future__ import annotations

from winnow import (
    NoEstimate,
    ReviewReason,
    SampleState,
    SampleStatus,
    StoppingCriterion,
)


def _state(
    *,
    samples: tuple[float, ...] = (),
    decline_count: int = 0,
    parse_failure_count: int = 0,
    consecutive_declines: int = 0,
    confidence: float = 0.0,
    status: SampleStatus = SampleStatus.COLLECTING,
    failure_reason: ReviewReason | None = None,
) -> SampleState[float]:
    return SampleState(
        samples=samples,
        decline_count=decline_count,
        parse_failure_count=parse_failure_count,
        consecutive_declines=consecutive_declines,
        current_estimate=samples[-1] if samples else NoEstimate,
        current_confidence=confidence,
        status=status,
        failure_reason=failure_reason,
    )


class TestStoppingCriterionEvaluate:
    def test_returns_none_while_collection_should_continue(self) -> None:
        criterion = StoppingCriterion(min_samples=2, max_queries=5)

        assert criterion.evaluate(_state(samples=(1.0,))) is None

    def test_convergence_wins_on_the_final_permitted_query(self) -> None:
        criterion = StoppingCriterion(
            min_samples=3,
            max_queries=3,
            confidence_threshold=0.9,
        )

        decision = criterion.evaluate(
            _state(samples=(1.0, 1.0, 1.0), confidence=1.0),
        )

        assert decision is not None
        assert decision.status is SampleStatus.CONVERGED
        assert decision.failure_reason is None

    def test_query_budget_without_minimum_samples_reports_max_queries(self) -> None:
        criterion = StoppingCriterion(min_samples=3, max_queries=5)

        decision = criterion.evaluate(
            _state(samples=(1.0, 2.0), decline_count=3),
        )

        assert decision is not None
        assert decision.failure_reason is ReviewReason.MAX_QUERIES

    def test_parse_failure_limit_wins_when_query_budget_is_also_exhausted(
        self,
    ) -> None:
        criterion = StoppingCriterion(min_samples=3, max_queries=5)

        decision = criterion.evaluate(
            _state(samples=(1.0, 2.0), parse_failure_count=3),
        )

        assert decision is not None
        assert decision.failure_reason is ReviewReason.MAX_PARSE_FAILURES

    def test_query_budget_after_minimum_samples_reports_low_confidence(self) -> None:
        criterion = StoppingCriterion(
            min_samples=3,
            max_queries=5,
            confidence_threshold=0.9,
        )

        decision = criterion.evaluate(
            _state(samples=(1.0, 2.0, 3.0), decline_count=2, confidence=0.4),
        )

        assert decision is not None
        assert decision.failure_reason is ReviewReason.INSUFFICIENT_CONFIDENCE

    def test_existing_terminal_decision_is_preserved(self) -> None:
        criterion = StoppingCriterion(min_samples=1, confidence_threshold=0.1)

        decision = criterion.evaluate(
            _state(
                samples=(1.0,),
                confidence=1.0,
                status=SampleStatus.NEEDS_REVIEW,
                failure_reason=ReviewReason.MAX_QUERIES,
            ),
        )

        assert decision is not None
        assert decision.status is SampleStatus.NEEDS_REVIEW
        assert decision.failure_reason is ReviewReason.MAX_QUERIES
