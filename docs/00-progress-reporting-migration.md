# Winnow Progress Reporting Migration

Status: Draft for review

## 1. Purpose

This series aligns Winnow with the progress-reporting API defined in `Winnow Docs`. The callback will receive one `CollectionProgress` value after each completed wave, containing the cumulative sample states and only the prompt-response interactions completed in that wave.

The migration also aligns the default confidence threshold with the documented value and updates the repository README.

## 2. Agreed Decisions

`CollectionProgress` and `QuestionInteraction` are supported public value types. They are re-exported through `winnow.__init__` and listed in `__all__`.

`QuestionUID`, `SampleStates` and `ProgressCallback` are supported public aliases. They name stable concepts in the collection contract rather than merely shortening incidental implementation annotations. `QuestionUID` is owned by `question.py`; the two progress-specific aliases are owned by `progress.py`.

`CollectionProgress.sample_states` is Winnow's current cumulative state mapping, not a historical snapshot. A caller that needs historical records must copy or persist the values during the callback.

`CollectionProgress.new_interactions` contains the completed prompt and raw response for every dispatched question, including declines and parse failures. Diagnostic JSONL logging remains optional and unchanged.

No compatibility adapter for the old two-argument callback is planned. Because the callback change is intentionally breaking, the migration bumps Winnow from `0.1.0` to `0.2.0` and makes the documented contract authoritative.

## 3. Target Ownership

```text
src/winnow/
├── __init__.py          # Public facade
├── collect.py           # Collection orchestration and callback emission
├── progress.py          # Public progress-reporting values
└── exchange/
    └── client.py        # Query execution, recording and completed interaction
```

`question.py` owns `QuestionUID` because it identifies the central question concept. `progress.py` owns the remaining progress values and aliases because they change with the progress-reporting contract. `ExchangeRecordingClient` constructs `QuestionInteraction` at the point where the complete prompt and raw response are both available. `collect()` processes those interactions and groups them into `CollectionProgress`.

## 4. Migration Sequence

The migration is divided into five independently reviewable phases:

1. `01-confidence-default.md` aligns the executable default with the documented `0.90`.
2. `02-progress-types-and-facade.md` adds the public progress values without changing callback behaviour.
3. `03-completed-exchange.md` carries complete interactions across the exchange boundary while preserving the old callback temporarily.
4. `04-collection-progress-callback.md` replaces the old callback contract with `CollectionProgress`.
5. `05-documentation-and-validation.md` updates the README and design notes, then validates the complete package.

Each phase must leave its relevant tests passing. Structural, exchange and public callback changes remain separate so each boundary can be reviewed in isolation.

## 5. Completion

The migration is complete when the package reports version `0.2.0`, the documented imports and callback example run against the installed package, each wave reports its cumulative states and new interactions, the README matches the executable facade, and all repository checks pass.
