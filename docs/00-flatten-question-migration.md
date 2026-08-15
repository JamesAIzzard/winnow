## Objective

Remove `Prompt[T]` and make `Question[T]` the single public definition of a typed sampling task. The refactor should simplify construction and reading without changing parsing, estimation, stopping, scheduling or progress behaviour.

The target public model is:

```python
@dataclass(frozen=True, kw_only=True)
class Question[T]:
    uid: QuestionUID
    query: str
    parser: Parser[T]
    estimator: Estimator[T]
    stopping_criterion: StoppingCriterion

    def build_prompt(self) -> str: ...
```

`QuestionUID` remains an opaque caller-supplied identifier. Winnow requires it to be unique within a `QuestionBank` and to match any supplied initial state. Stability across collection calls or application runs is caller policy.

`build_prompt()` continues to append Winnow's decline instruction to `query`. It belongs on `Question` because that instruction is coupled to decline parsing and stopping behaviour.

## Scope

The migration removes:

- the `Prompt[T]` class;
- `Question.prompt` and its delegating properties;
- `Prompt` from the top-level facade;
- nested `Question(prompt=Prompt(...))` construction;
- exchange code that accepts a prompt object rather than the selected question.

The migration does not:

- change parser, estimator or stopping semantics;
- change wave selection, concurrency or progress reporting;
- prescribe whether UIDs remain stable outside one collection call;
- promise caller-side reuse of `question.parser`;
- retain a compatibility wrapper in the final API.

## Phase 1: Flatten Winnow

Change the model, its internal consumers and its tests together so the repository remains coherent.

- Move `uid`, `query`, `parser` and `build_prompt()` onto `Question[T]` in `question.py`.
- Delete `Prompt[T]` and the delegating `Question` properties.
- Change `ExchangeRecordingClient` to accept a `Question[T]`, build the transmitted prompt from it, and record the same `QuestionInteraction` values as before.
- Change `collect()` to pass each selected question directly to the exchange client.
- Remove `Prompt` from `winnow.__init__` and `__all__`.
- Replace test fixtures and helpers that construct `Prompt[T]` with flat `Question[T]` construction.
- Move prompt-building assertions to `Question.build_prompt()`.
- Preserve tests for the exact query text, decline instruction, question UID and recorded raw response.
- Preserve collection tests for estimates, review outcomes, initial states, wave scheduling and progress callbacks.
- Add a facade assertion that `Question` remains public and `Prompt` is absent.
- Keep `QuestionBank`, `SampleState` and result behaviour unchanged.

This phase is complete when the full Winnow suite passes and the runtime facade no longer exposes `Prompt`.

## Phase 2: Clean the Winnow Surface

Make every repository-facing description and example match the implemented API.

- Update `README.md` imports, examples and explanatory prose.
- Remove obsolete prompt-specific test helpers and names.
- Search `src`, `tests` and `README.md` for remaining `Prompt` type references and nested construction.
- Bump the package minor version because removing a public type is a breaking change while the project is pre-1.0.
- Run the full test, lint and type-check suite.

Verification:

```powershell
uv run pytest
uv run ruff check .
uv run mypy src
uv run pyright
```

This phase is complete when Winnow is internally consistent, fully validated and ready for downstream integration testing.

## Phase 3: Migrate Dietrix

Update Dietrix against the refactor branch or its exact commit before releasing Winnow.

- Flatten every `Question(...)` construction in `codiet-data-pop`.
- Remove `Prompt` imports and the `to_winnow_prompt()` adapter.
- Update `SampledField.to_winnow_question()` to populate all `Question[T]` fields directly.
- Update reference-quantity and grams-per-reference-quantity factories to construct flat questions.
- Update the Winnow compatibility tests to exercise the new facade and progress contract.
- Keep Dietrix-owned `FieldPrompt` and human-response parsing separate from Winnow. Do not replace the removed Winnow type with another wrapper solely to preserve the old shape.
- Adapt single-shot LLM prompt construction explicitly at the Dietrix boundary. It may use a complete flat question where one already exists, but Winnow should not add a public abstraction solely for this caller path.

Verification should include the focused compatibility tests followed by the relevant `codiet-data-pop` test, lint and type-check suites.

This phase is complete when Dietrix imports no Winnow `Prompt`, passes its compatibility checks, and consumes the refactored Winnow commit without a local shim.

## Phase 4: Release and Cut Over

Release only after Winnow and Dietrix pass together.

- Review the final public diff for accidental behaviour changes.
- Commit and push the Winnow refactor branch.
- Merge and tag the new Winnow version.
- Replace Dietrix's temporary branch or commit dependency with the released tag and refresh `uv.lock`.
- Run Dietrix's final compatibility check against the released artefact.

The migration is complete when the released Winnow facade exposes only the flat `Question[T]` model and Dietrix is locked to that release.
