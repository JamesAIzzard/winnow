# Consensus

**Statistically robust data extraction from large language models.**

Consensus is a Python library for confidence-quantified data extraction through repeated sampling and type-aware aggregation. It treats the LLM as a stochastic oracle, collecting multiple responses to a battery of questions and deriving point estimates with explicit confidence scores.

## The Problem

Large language models are stochastic. Ask the same question twice and you may receive different answers. For many applications—chatbots, creative writing, code generation—this variability is acceptable or even desirable. But when extracting structured data that will feed into downstream systems, we need to know whether the model is confident or guessing.

Consider querying an LLM for the protein content of chicken breast. Across ten queries, you might receive: 31g, 31g, 29g, 31g, 280g, 30g, 31g, 32g, 31g, 30g. Nine responses cluster tightly around 31g; one is wildly wrong. A naive approach that takes a single sample has a 10% chance of returning 280g. Taking the mean yields 80.6g—worse than random selection. The correct approach is to recognise that the median (31g) represents the model's consistent belief, the spread is low, and the outlier should be discarded.

This statistical treatment is straightforward for numerical data. But LLM extraction involves diverse data types: continuous values, categorical selections, boolean flags, and more. Each requires a different consensus strategy. Existing libraries handle parsing and validation; none handle the statistical layer that sits above it.

## Core Concepts

### The Stochastic Oracle Model

Consensus treats the LLM as a *stochastic oracle*: a black box that, when queried, returns samples from some underlying distribution. The oracle may be noisy (high variance), biased (systematic error), or unreliable (frequent refusals). Our goal is to characterise this distribution efficiently and extract a point estimate with quantified confidence.

This abstraction is deliberately general. While the primary use case is LLM extraction, the same framework applies to any repeated-query scenario: crowdsourced labelling, sensor fusion, or aggregating responses from multiple models.

### Confidence, Not Certainty

Consensus produces a confidence score in the range [0, 1] for every extracted value. This score reflects the *consistency* of the oracle's responses, not the *correctness* of the answer. An LLM might consistently hallucinate that tomatoes contain 40g of protein; Consensus would report high confidence in this wrong answer.

This is a feature, not a limitation. Confidence scores identify *where the model is uncertain*, allowing downstream systems to flag values for review, apply wider error bounds, or fall back to authoritative sources. Detecting consistent-but-wrong answers requires domain-specific invariant checking, which is outside the scope of Consensus but can be layered on top.

### Type-Aware Aggregation

Different data types demand different consensus strategies:

| Data Type | Estimate | Confidence Basis |
|-----------|----------|------------------|
| Continuous numerical | Median | Robust coefficient of variation |
| Categorical (from known set) | Mode | Agreement proportion (normalised) |
| Boolean | Mode | Agreement proportion |
| Ordinal | Median | Rank-based spread |

Consensus provides built-in estimators for common types and an extension point for custom aggregation logic.

### Question-Centric Design

Rather than extracting complex structured objects, Consensus operates on a *battery of questions*. Each question is a focused query with its own parser, estimator, and stopping criterion. This design provides:

- **Explicit control over prompts.** Each question is a carefully crafted query string, not a schema being converted to a prompt behind the scenes.
- **Per-question confidence thresholds.** Critical fields can demand higher confidence than trace data.
- **Natural decline handling.** The model can opt out of answering using explicit keywords.
- **Randomised sampling.** When multiple questions are pending, the system interleaves them to avoid response "ruts" where the model anchors on its initial answer.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                      Consensus                          │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐  │
│  │ QuestionBank  │  │  Estimators   │  │  Stopping   │  │
│  │               │  │               │  │  Criteria   │  │
│  │ • Questions   │  │ • Numerical   │  │             │  │
│  │ • Selection   │  │ • Categorical │  │ • Composable│  │
│  │ • Randomising │  │ • Boolean     │  │ • Per-field │  │
│  └───────────────┘  └───────────────┘  └─────────────┘  │
│  ┌───────────────┐  ┌───────────────┐                   │
│  │   Parsers     │  │   Sampling    │                   │
│  │               │  │               │                   │
│  │ • Float       │  │ • State mgmt  │                   │
│  │ • Literal     │  │ • Decline     │                   │
│  │ • Boolean     │  │   tracking    │                   │
│  │ • Custom      │  │ • Parallelism │                   │
│  └───────────────┘  └───────────────┘                   │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     LLM Provider                        │
│            OpenAI • Anthropic • Ollama • etc.           │
└─────────────────────────────────────────────────────────┘
```

## Core Types

### Parser

A `Parser` converts raw LLM response strings into typed values. It handles both successful parsing and explicit declines from the model.

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')


class ParserError(Exception):
    """Raised when a response cannot be parsed into the expected type."""
    pass


class Parser(ABC, Generic[T]):
    """Converts raw LLM responses into typed values."""
    
    decline_keywords: frozenset[str] = frozenset({"UNKNOWN", "INSUFFICIENT_DATA"})
    
    def __call__(self, response: str) -> T | None:
        """
        Parse a response string.
        
        Returns:
            The parsed value, or None if the model declined.
        
        Raises:
            ParserError: If the response cannot be parsed.
        """
        normalised = response.strip().upper()
        if any(keyword in normalised for keyword in self.decline_keywords):
            return None
        return self.parse(response)
    
    @abstractmethod
    def parse(self, response: str) -> T:
        """
        Parse a non-decline response into the target type.
        
        Raises:
            ParserError: If parsing fails.
        """
        ...
```

The explicit `decline_keywords` mechanism allows the model to opt out gracefully. A response containing "UNKNOWN" or "INSUFFICIENT_DATA" is treated as a decline rather than a parse failure, and these are tracked separately in the sampling state.

### Estimator

An `Estimator` implements type-specific aggregation, computing both a point estimate and a confidence score from collected samples.

```python
from typing import Protocol, Sequence

class ConsensusEstimator(Protocol[T]):
    """Strategy for deriving consensus from repeated samples."""
    
    def compute_estimate(self, samples: Sequence[T]) -> T:
        """Derive the best point estimate from collected samples."""
        ...
    
    def compute_confidence(self, samples: Sequence[T], estimate: T) -> float:
        """Compute confidence in the estimate, normalised to [0, 1]."""
        ...
```

### Stopping Criterion

A `StoppingCriterion` determines when sampling should stop for a question. Criteria are composable using `&` (all must agree) and `|` (any can trigger).

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SampleState(Generic[T]):
    """Current sampling state for a single question."""
    samples: tuple[T, ...]
    decline_count: int
    parse_failure_count: int
    consecutive_declines: int
    
    @property
    def query_count(self) -> int:
        return len(self.samples) + self.decline_count + self.parse_failure_count


class StoppingCriterion(ABC):
    """Determines when sampling should stop for a question."""
    
    @abstractmethod
    def should_stop(
        self, 
        state: SampleState[T], 
        estimator: ConsensusEstimator[T],
    ) -> bool:
        """Return True if sampling should stop."""
        ...
    
    def __and__(self, other: StoppingCriterion) -> StoppingCriterion:
        """Both criteria must agree to stop."""
        return All(self, other)
    
    def __or__(self, other: StoppingCriterion) -> StoppingCriterion:
        """Either criterion can trigger a stop."""
        return Any(self, other)
```

### Question

A `Question` bundles a query string with its parser, estimator, and stopping criterion.

```python
@dataclass(frozen=True)
class Question(Generic[T]):
    """A query paired with its parsing and estimation strategy."""
    
    uid: str                                # Unique identifier
    query: str                              # The prompt to send to the LLM
    parser: Parser[T]                       # Converts response to typed value
    estimator: ConsensusEstimator[T]        # Aggregation strategy
    stopping_criterion: StoppingCriterion   # When to stop sampling
```

### Estimate

An `Estimate` is the output for a single question, containing the value and all metadata about how it was derived.

```python
from enum import Enum, auto


class Archetype(Enum):
    """Classification of sampling convergence behaviour."""
    CONFIDENT = auto()         # High confidence, stopped early
    ACCEPTABLE = auto()        # Met threshold
    UNCERTAIN = auto()         # Below threshold, budget exhausted
    INSUFFICIENT_DATA = auto() # Too many declines to form estimate


@dataclass(frozen=True)
class Estimate(Generic[T]):
    """A value estimated from repeated LLM queries."""
    
    value: T
    confidence: float
    archetype: Archetype
    sample_count: int
    decline_count: int
    samples: tuple[T, ...]  # Raw samples, for diagnostics
```

### QuestionBank

A `QuestionBank` holds the collection of questions and manages selection during sampling.

```python
from typing import Sequence
import random


@dataclass
class QuestionBank:
    """A collection of questions to be answered."""
    
    questions: Sequence[Question]
    
    def select_next(
        self, 
        states: dict[str, SampleState],
    ) -> Question | None:
        """
        Select the next question to ask.
        
        Returns an incomplete question at random, or None if all complete.
        Randomisation prevents the model from anchoring on repeated queries.
        """
        incomplete = [
            q for q in self.questions
            if not q.stopping_criterion.should_stop(states[q.uid], q.estimator)
        ]
        
        if not incomplete:
            return None
        
        return random.choice(incomplete)
```

## Built-in Components

### Parsers

```python
class FloatParser(Parser[float]):
    """Parses a floating-point number, optionally with unit conversion."""
    
    def __init__(
        self, 
        expected_unit: str | None = None,
        unit_conversions: dict[str, float] | None = None,
    ):
        self.expected_unit = expected_unit
        self.unit_conversions = unit_conversions or {}
    
    def parse(self, response: str) -> float:
        match = re.search(r'([\d.]+)\s*(\w*)', response.strip())
        if not match:
            raise ParserError(f"Could not extract number from: {response}")
        
        value = float(match.group(1))
        unit = match.group(2).lower() if match.group(2) else None
        
        if unit and unit in self.unit_conversions:
            value *= self.unit_conversions[unit]
        
        return value


class LiteralParser(Parser[T], Generic[T]):
    """Parses a response matching one of a known set of values."""
    
    def __init__(
        self, 
        options: frozenset[T], 
        case_sensitive: bool = False,
    ):
        self.options = options
        self.case_sensitive = case_sensitive
        self._lookup = {
            (str(o) if case_sensitive else str(o).lower()): o 
            for o in options
        }
    
    def parse(self, response: str) -> T:
        key = response.strip() if self.case_sensitive else response.strip().lower()
        if key not in self._lookup:
            raise ParserError(
                f"Response '{response}' not in valid options: {self.options}"
            )
        return self._lookup[key]


class BooleanParser(Parser[bool]):
    """Parses yes/no, true/false style responses."""
    
    truthy: frozenset[str] = frozenset({"yes", "true", "1", "y"})
    falsy: frozenset[str] = frozenset({"no", "false", "0", "n"})
    
    def parse(self, response: str) -> bool:
        normalised = response.strip().lower()
        if normalised in self.truthy:
            return True
        if normalised in self.falsy:
            return False
        raise ParserError(f"Could not parse boolean from: {response}")
```

### Estimators

```python
class NumericalEstimator(ConsensusEstimator[float]):
    """Consensus estimation for continuous numerical values."""
    
    def compute_estimate(self, samples: Sequence[float]) -> float:
        return _median(samples)
    
    def compute_confidence(self, samples: Sequence[float], estimate: float) -> float:
        if len(samples) < 2:
            return 0.0
        
        if all(s == 0.0 for s in samples):
            return 1.0  # Unanimous zero is high confidence
        
        if estimate == 0.0:
            return 0.0  # Non-zero samples but zero median
        
        mad = _median([abs(s - estimate) for s in samples])
        robust_cv = 1.4826 * mad / abs(estimate)
        
        return 1.0 / (1.0 + robust_cv)


class CategoricalEstimator(ConsensusEstimator[T], Generic[T]):
    """Consensus estimation for categorical values."""
    
    def __init__(self, valid_options: frozenset[T]):
        self.valid_options = valid_options
    
    def compute_estimate(self, samples: Sequence[T]) -> T:
        counts = Counter(samples)
        return counts.most_common(1)[0][0]
    
    def compute_confidence(self, samples: Sequence[T], estimate: T) -> float:
        if len(samples) == 0:
            return 0.0
        
        agreement = sum(1 for s in samples if s == estimate) / len(samples)
        baseline = 1.0 / len(self.valid_options)
        
        if baseline >= 1.0:
            return 1.0
        
        return (agreement - baseline) / (1.0 - baseline)


class BooleanEstimator(ConsensusEstimator[bool]):
    """Consensus estimation for boolean values."""
    
    def compute_estimate(self, samples: Sequence[bool]) -> bool:
        return sum(samples) > len(samples) / 2
    
    def compute_confidence(self, samples: Sequence[bool], estimate: bool) -> float:
        if len(samples) == 0:
            return 0.0
        
        agreement = sum(1 for s in samples if s == estimate) / len(samples)
        return agreement  # Boolean baseline is 0.5, but raw agreement is intuitive
```

### Stopping Criteria

**Primitive criteria:**

```python
@dataclass(frozen=True)
class MinSamples(StoppingCriterion):
    """Don't stop until we have at least n successful samples."""
    n: int
    
    def should_stop(self, state: SampleState, estimator: ConsensusEstimator) -> bool:
        return len(state.samples) >= self.n


@dataclass(frozen=True)
class MaxQueries(StoppingCriterion):
    """Stop after n total queries (success + decline + failure)."""
    n: int
    
    def should_stop(self, state: SampleState, estimator: ConsensusEstimator) -> bool:
        return state.query_count >= self.n


@dataclass(frozen=True)
class ConfidenceReached(StoppingCriterion):
    """Stop when confidence exceeds threshold."""
    threshold: float
    
    def should_stop(self, state: SampleState, estimator: ConsensusEstimator) -> bool:
        if len(state.samples) < 2:
            return False
        estimate = estimator.compute_estimate(state.samples)
        confidence = estimator.compute_confidence(state.samples, estimate)
        return confidence >= self.threshold


@dataclass(frozen=True)
class ConsecutiveDeclines(StoppingCriterion):
    """Stop if model declines n times in a row."""
    n: int
    
    def should_stop(self, state: SampleState, estimator: ConsensusEstimator) -> bool:
        return state.consecutive_declines >= self.n


@dataclass(frozen=True)
class UnanimousAgreement(StoppingCriterion):
    """Stop early if all samples agree (useful for categorical/boolean)."""
    min_samples: int = 3
    
    def should_stop(self, state: SampleState, estimator: ConsensusEstimator) -> bool:
        if len(state.samples) < self.min_samples:
            return False
        return len(set(state.samples)) == 1
```

**Combinators:**

```python
@dataclass(frozen=True)
class All(StoppingCriterion):
    """Stop only when all child criteria agree."""
    criteria: tuple[StoppingCriterion, ...]
    
    def __init__(self, *criteria: StoppingCriterion):
        object.__setattr__(self, 'criteria', criteria)
    
    def should_stop(self, state: SampleState, estimator: ConsensusEstimator) -> bool:
        return all(c.should_stop(state, estimator) for c in self.criteria)


@dataclass(frozen=True)
class Any(StoppingCriterion):
    """Stop when any child criterion is satisfied."""
    criteria: tuple[StoppingCriterion, ...]
    
    def __init__(self, *criteria: StoppingCriterion):
        object.__setattr__(self, 'criteria', criteria)
    
    def should_stop(self, state: SampleState, estimator: ConsensusEstimator) -> bool:
        return any(c.should_stop(state, estimator) for c in self.criteria)
```

**Convenience factories:**

```python
def standard_stopping(
    min_samples: int = 5,
    confidence: float = 0.90,
    max_queries: int = 20,
    max_consecutive_declines: int = 5,
) -> StoppingCriterion:
    """Standard stopping criterion for numerical fields."""
    return (
        (MinSamples(min_samples) & ConfidenceReached(confidence)) 
        | MaxQueries(max_queries) 
        | ConsecutiveDeclines(max_consecutive_declines)
    )


def categorical_stopping(
    unanimous_after: int = 3,
    min_samples: int = 5,
    confidence: float = 0.85,
    max_queries: int = 15,
) -> StoppingCriterion:
    """Stopping criterion for categorical fields with early unanimous exit."""
    return (
        UnanimousAgreement(unanimous_after) 
        | (MinSamples(min_samples) & ConfidenceReached(confidence)) 
        | MaxQueries(max_queries)
    )


def relaxed_stopping(
    min_samples: int = 5,
    confidence: float = 0.75,
    max_queries: int = 15,
    max_consecutive_declines: int = 3,
) -> StoppingCriterion:
    """Relaxed criterion for inherently variable data (e.g., trace nutrients)."""
    return (
        (MinSamples(min_samples) & ConfidenceReached(confidence)) 
        | MaxQueries(max_queries) 
        | ConsecutiveDeclines(max_consecutive_declines)
    )
```

## The Collection Function

The main entry point is the `collect` function, which processes a `QuestionBank` and returns estimates for all questions.

```python
from typing import Callable, Awaitable


async def collect(
    bank: QuestionBank,
    query_fn: Callable[[str], Awaitable[str]],
) -> dict[str, Estimate]:
    """
    Collect estimates for all questions in the bank.
    
    Args:
        bank: The questions to answer.
        query_fn: Async function that sends a query string to the LLM
                  and returns the raw response string.
    
    Returns:
        Mapping from question UID to its estimate.
    """
    states: dict[str, SampleState] = {
        q.uid: SampleState(
            samples=(),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
        )
        for q in bank.questions
    }
    
    while (question := bank.select_next(states)) is not None:
        response = await query_fn(question.query)
        
        try:
            result = question.parser(response)
            if result is None:
                states[question.uid] = _record_decline(states[question.uid])
            else:
                states[question.uid] = _record_sample(states[question.uid], result)
        except ParserError:
            states[question.uid] = _record_parse_failure(states[question.uid])
    
    return _build_estimates(bank.questions, states)


def _record_sample(state: SampleState[T], value: T) -> SampleState[T]:
    return SampleState(
        samples=state.samples + (value,),
        decline_count=state.decline_count,
        parse_failure_count=state.parse_failure_count,
        consecutive_declines=0,  # Reset on successful sample
    )


def _record_decline(state: SampleState[T]) -> SampleState[T]:
    return SampleState(
        samples=state.samples,
        decline_count=state.decline_count + 1,
        parse_failure_count=state.parse_failure_count,
        consecutive_declines=state.consecutive_declines + 1,
    )


def _record_parse_failure(state: SampleState[T]) -> SampleState[T]:
    return SampleState(
        samples=state.samples,
        decline_count=state.decline_count,
        parse_failure_count=state.parse_failure_count + 1,
        consecutive_declines=0,  # Parse failure breaks decline streak
    )


def _build_estimates(
    questions: Sequence[Question],
    states: dict[str, SampleState],
) -> dict[str, Estimate]:
    estimates = {}
    
    for q in questions:
        state = states[q.uid]
        
        if len(state.samples) == 0:
            estimates[q.uid] = Estimate(
                value=None,
                confidence=0.0,
                archetype=Archetype.INSUFFICIENT_DATA,
                sample_count=0,
                decline_count=state.decline_count,
                samples=(),
            )
            continue
        
        value = q.estimator.compute_estimate(state.samples)
        raw_confidence = q.estimator.compute_confidence(state.samples, value)
        
        # Adjust confidence for decline rate
        total_attempts = len(state.samples) + state.decline_count
        decline_penalty = 1.0 - (state.decline_count / total_attempts) if total_attempts > 0 else 1.0
        confidence = raw_confidence * decline_penalty
        
        # Determine archetype
        archetype = _classify_archetype(q, state, confidence)
        
        estimates[q.uid] = Estimate(
            value=value,
            confidence=confidence,
            archetype=archetype,
            sample_count=len(state.samples),
            decline_count=state.decline_count,
            samples=state.samples,
        )
    
    return estimates
```

## Example Usage

### Nutritional Data Collection

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()


async def query_llm(prompt: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "Answer concisely. If uncertain, reply UNKNOWN."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


ingredient = "chicken breast"

questions = QuestionBank([
    Question(
        uid="protein",
        query=f"How many grams of protein per 100g of {ingredient}? Reply with just the number.",
        parser=FloatParser(),
        estimator=NumericalEstimator(),
        stopping_criterion=standard_stopping(confidence=0.90),
    ),
    Question(
        uid="fat",
        query=f"How many grams of fat per 100g of {ingredient}? Reply with just the number.",
        parser=FloatParser(),
        estimator=NumericalEstimator(),
        stopping_criterion=standard_stopping(confidence=0.90),
    ),
    Question(
        uid="carbohydrate",
        query=f"How many grams of carbohydrate per 100g of {ingredient}? Reply with just the number.",
        parser=FloatParser(),
        estimator=NumericalEstimator(),
        stopping_criterion=standard_stopping(confidence=0.90),
    ),
    Question(
        uid="is_vegan",
        query=f"Is {ingredient} vegan? Reply YES, NO, or UNKNOWN.",
        parser=BooleanParser(),
        estimator=BooleanEstimator(),
        stopping_criterion=categorical_stopping(),
    ),
    Question(
        uid="standard_unit",
        query=f"What unit is typically used to measure {ingredient}: gram, piece, breast, or cup? Reply with just the unit, or UNKNOWN.",
        parser=LiteralParser(frozenset({"gram", "piece", "breast", "cup"})),
        estimator=CategoricalEstimator(frozenset({"gram", "piece", "breast", "cup"})),
        stopping_criterion=categorical_stopping(),
    ),
    Question(
        uid="selenium",
        query=f"How many micrograms of selenium per 100g of {ingredient}? Reply with just the number, or UNKNOWN.",
        parser=FloatParser(),
        estimator=NumericalEstimator(),
        stopping_criterion=relaxed_stopping(confidence=0.75),  # Trace nutrient
    ),
])

estimates = asyncio.run(collect(questions, query_llm))

for uid, est in estimates.items():
    print(f"{uid}: {est.value} (confidence={est.confidence:.2f}, n={est.sample_count})")
```

**Example output:**

```
protein: 31.0 (confidence=0.94, n=7)
fat: 3.6 (confidence=0.91, n=8)
carbohydrate: 0.0 (confidence=1.00, n=5)
is_vegan: False (confidence=1.00, n=3)
standard_unit: breast (confidence=0.88, n=6)
selenium: 27.5 (confidence=0.78, n=12)
```

## Implementation Details

### Numerical Confidence

For continuous numerical values, we use the median as the point estimate and the robust coefficient of variation as the confidence basis:

$$r_{st} = \frac{1.4826 \cdot \text{MAD}}{\tilde{x}}$$

> Where:
> $\text{MAD}$ is the median absolute deviation
> $\tilde{x}$ is the median
> The constant 1.4826 scales MAD to be comparable to standard deviation for normally distributed data

Confidence is then:

$$\text{confidence} = \frac{1}{1 + r_{st}}$$

This maps $r_{st} \in [0, \infty)$ to confidence $\in (0, 1]$, with a threshold of $r_{st} = 0.10$ corresponding to confidence $\approx 0.91$.

### Categorical Confidence

For categorical values drawn from a known set, we use the mode as the point estimate and normalised agreement as the confidence basis:

$$\text{confidence} = \frac{p - \frac{1}{n}}{1 - \frac{1}{n}}$$

> Where:
> $p$ is the proportion of samples matching the mode
> $n$ is the number of valid options

This normalises against random guessing: if there are 5 options, random selection yields 20% agreement, so we scale confidence relative to this baseline.

### Decline Penalty

The confidence score incorporates decline rate to penalise estimates where the model frequently refused to answer:

$$\text{confidence}_{\text{adjusted}} = \text{confidence}_{\text{raw}} \cdot \left(1 - \frac{d}{d + s}\right)$$

> Where:
> $d$ is the decline count
> $s$ is the successful sample count

### Randomised Selection

When multiple questions remain incomplete, `QuestionBank.select_next()` chooses randomly among them. This prevents the model from anchoring on repeated identical queries, which empirically causes responses to cluster more tightly than they should—a phenomenon where the model gets "stuck" in a response pattern.

## Project Structure

```
consensus/
├── __init__.py
├── collect.py          # collect() function, main entry point
├── types.py            # Estimate, Archetype, SampleState
├── question.py         # Question, QuestionBank
├── parser/
│   ├── __init__.py
│   ├── base.py         # Parser ABC, ParserError
│   ├── numerical.py    # FloatParser, IntParser
│   ├── categorical.py  # LiteralParser
│   └── boolean.py      # BooleanParser
├── estimator/
│   ├── __init__.py
│   ├── base.py         # ConsensusEstimator protocol
│   ├── numerical.py    # NumericalEstimator
│   ├── categorical.py  # CategoricalEstimator
│   └── boolean.py      # BooleanEstimator
├── stopping/
│   ├── __init__.py
│   ├── base.py         # StoppingCriterion ABC
│   ├── primitives.py   # MinSamples, MaxQueries, etc.
│   ├── combinators.py  # All, Any
│   └── factories.py    # standard_stopping, categorical_stopping, etc.
└── _util.py            # _median and other internal helpers
```

## Development Roadmap

### Phase 1: Core Infrastructure

- Core types: `Estimate`, `Archetype`, `SampleState`
- `Parser` ABC and built-in parsers: `FloatParser`, `LiteralParser`, `BooleanParser`
- `ConsensusEstimator` protocol and built-in estimators
- `StoppingCriterion` ABC, primitives, and combinators
- `Question` and `QuestionBank` types
- `collect()` function with sequential execution

### Phase 2: Robustness

- Comprehensive test suite with synthetic data
- Edge case handling: empty samples, all declines, unanimous agreement
- Logging and diagnostics
- Parse failure retry logic (optional retry before recording failure)

### Phase 3: Performance

- Parallel query execution with configurable concurrency
- Rate limit handling
- Caching layer for repeated queries across ingredients
- Progress callbacks for long-running collections

### Phase 4: Extensions

- Ordinal estimator for ranked data
- Custom parser composition utilities
- Multi-model consensus (query multiple LLMs, aggregate across models)
- Integration with domain-specific validation (e.g., nutrient tree balancing)

## Testing Strategy

**Unit tests** verify parsers, estimators, and stopping criteria using synthetic data. These are fast, deterministic, and exercise edge cases.

**Property-based tests** (using Hypothesis) verify that estimators behave correctly across random distributions—e.g., that confidence increases with sample agreement, that the median is robust to outliers.

**Integration tests** use a mock `query_fn` that returns predetermined sequences, verifying that the sampling loop behaves correctly under various response patterns.

**Characterisation tests** run against real LLMs with known-answer queries to verify that confidence scores correlate with actual accuracy. These run on a schedule rather than per-commit.

## Dependencies

**Required:**
- Python ≥ 3.11 (for `Self` type hints and modern syntax)

**Optional:**
- `numpy` for numerical operations (falls back to pure Python)

## Prior Art and Differentiation

**Instructor** provides structured extraction with validation and retries. It operates on a single-response model and does not quantify confidence.

**Self-consistency research** (Wang et al., 2022) uses majority voting to improve reasoning accuracy. This assumes a discrete answer space and external verification.

**CISC** (Confidence Improves Self-Consistency) adds confidence-weighted voting to self-consistency. It remains focused on reasoning tasks and does not provide a reusable library.

**Consensus** differs in treating the problem as statistical estimation over a stochastic oracle, with type-aware aggregation, per-question stopping criteria, and explicit confidence scores attached to every output.

## Conclusion

Consensus addresses a gap in the current LLM tooling ecosystem: the need for confidence-quantified data extraction. By treating the LLM as a stochastic oracle and applying type-aware statistical aggregation with composable stopping criteria, it enables applications that require reliable structured data rather than best-effort responses.

The question-centric design provides explicit control over prompts and per-field confidence requirements, whilst the randomised sampling strategy avoids the response anchoring that plagues sequential identical queries. The result is a focused library that does one thing well: extract data from LLMs with quantified confidence.