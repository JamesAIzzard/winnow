# Winnow

Statistically robust data extraction from large language models.

Winnow treats an LLM as a stochastic oracle. It repeatedly asks the same questions and applies statistical estimation until each answer converges or requires review.

## Usage

```python
import asyncio

from winnow import (
    Estimate,
    FloatParser,
    NeedsReview,
    NumericalEstimator,
    Question,
    QuestionBank,
    StoppingCriterion,
    collect,
)


async def query_llm(prompt: str) -> str:
    response = await your_llm_client.query(prompt)
    return response


async def main() -> None:
    bank = QuestionBank([
        Question(
            uid="protein",
            query="How many grams of protein are in 100g of chicken breast?",
            parser=FloatParser(),
            estimator=NumericalEstimator(),
            stopping_criterion=StoppingCriterion(),
        ),
    ])

    results = await collect(bank=bank, query_fn=query_llm)
    result = results["protein"]

    if isinstance(result, Estimate):
        print(f"Protein: {result.value}g ({result.confidence:.0%} confidence)")
    elif isinstance(result, NeedsReview):
        print(f"Protein needs review: {result.reason.value}")


asyncio.run(main())
```

Each `Question` owns its uid, query, parser, estimator and `StoppingCriterion`. `collect()` returns an `Estimate` when sampling converges or `NeedsReview` when it stops without sufficient confidence.

## Available Components

| Parser | Estimator | Use case |
| --- | --- | --- |
| `FloatParser` | `NumericalEstimator` | Numerical values using the median |
| `BooleanParser` | `BooleanEstimator` | Boolean values using majority agreement |
| `LiteralParser` | `CategoricalEstimator` | Values from a fixed set of options |
| `FloatLiteralPairParser` | `OpenCategoricalEstimator` | Compound numerical and categorical values |
| `OptionalBoundedIntParser` | `OptionalIntEstimator` | Bounded integers that may not apply |

## Progress Reporting

`on_progress` is called once after each completed wave:

```python
from winnow import CollectionProgress, NoEstimate


def show_progress(progress: CollectionProgress) -> None:
    for interaction in progress.new_interactions:
        print(f"{interaction.question_uid}: {interaction.raw_response}")

    for question_uid, state in progress.sample_states.items():
        if state.current_estimate is not NoEstimate:
            print(
                f"{question_uid}: {state.current_estimate} "
                f"({state.current_confidence:.0%})"
            )


results = await collect(
    bank=bank,
    query_fn=query_llm,
    on_progress=show_progress,
)
```

`sample_states` contains the current cumulative state of every question. `new_interactions` contains only the prompts and raw responses completed in that wave, including responses that were declined or could not be parsed. Winnow does not retain progress history for the caller; copy or persist the values during the callback when history is required.
