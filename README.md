# Winnow

Statistically robust data extraction from large language models.

## Simple Usage Example

Winnow treats an LLM as a stochastic oracle—repeatedly querying it and using statistical methods to derive confident estimates.

```python
import asyncio
from winnow import (
    Question,
    QuestionBank,
    collect,
    FloatParser,
    NumericalEstimator,
    StoppingCriterion,
)


async def main():
    # Define your LLM query function
    async def query_llm(prompt: str) -> str:
        # Replace with your actual LLM call
        response = await your_llm_client.query(prompt)
        return response

    # Create a bank of questions to ask
    bank = QuestionBank([
        Question(
            uid="protein",
            query="How many grams of protein are in 100g of chicken breast?",
            parser=FloatParser(),
            estimator=NumericalEstimator(),
            stopping_criterion=StoppingCriterion(min_samples=5),
        ),
    ])

    # Collect estimates by repeatedly querying the LLM
    results = await collect(bank=bank, query_fn=query_llm)

    # Each result contains a value and confidence score
    estimate = results["protein"]
    print(f"Protein: {estimate.value}g (confidence: {estimate.confidence:.0%})")


asyncio.run(main())
```

### How It Works

1. **Define questions** with a unique ID, prompt text, parser, and estimator
2. **Provide a query function** that sends prompts to your LLM
3. **Call `collect()`** which repeatedly queries the LLM until confidence thresholds are met
4. **Use the results** — each estimate includes both a value and a confidence score

### Available Components

| Parsers | Estimators | Use Case |
|---------|------------|----------|
| `FloatParser` | `NumericalEstimator` | Numeric values (uses median) |
| `BooleanParser` | `BooleanEstimator` | Yes/no questions (uses majority vote) |
| `LiteralParser` | `CategoricalEstimator` | Fixed set of options (uses mode) |

### Handling Uncertainty

If the LLM's responses are inconsistent, the confidence score will be low. You can use this to flag results for human review:

```python
for uid, estimate in results.items():
    if estimate.confidence < 0.8:
        print(f"Warning: {uid} has low confidence ({estimate.confidence:.0%})")
```

### Progress Tracking

You can track progress during collection using the `on_progress` callback:

```python
from winnow import collect, SampleState

def show_progress(states: dict[str, SampleState]) -> None:
    for uid, state in states.items():
        if state.current_estimate is not None:
            print(f"{uid}: {state.current_estimate} ({state.current_confidence:.0%})")

results = await collect(
    bank=bank,
    query_fn=query_llm,
    on_progress=show_progress,
)
```

The callback receives a dictionary of `SampleState` objects, each containing:
- `current_estimate`: The current best estimate (or `None` if no samples yet)
- `current_confidence`: The current confidence level (0.0 to 1.0)
- `samples`: All collected samples so far
- `query_count`: Total queries made (including declines and parse failures)
