# Winnow Structure Migration

Status: Draft for review

## 1. Purpose

This series plans the migration from Winnow's current repository structure to the structure defined in the Lodestone notes `Winnow Project Structure` and `Winnow Docs`.

The target is a small Python library with one public facade at `winnow/__init__.py`. Root modules retain the central sampling model. The `parser`, `estimator` and `exchange` packages own cohesive implementation families or a clear external boundary.

The migration is deliberately narrow. It does not redesign Winnow's domain model, introduce an API object, add dependency injection, or create a generic `core`, `common`, `helpers` or `utils` package.

## 2. Agreed Decisions

`LoggedLLMClient` and `log_exchange` are implementation details and will be removed from the public API. `LLMClient` remains the public caller-supplied query protocol, and `configure_jsonl_logging` remains a public stateless convenience function.

The documented top-level imports remain stable. Provider clients, credentials, retry policies and source-specific configuration continue to belong to the caller.

Internal modules will use relative imports. Tests will prefer the top-level facade where they exercise public behaviour and use internal imports only for focused parser, estimator and exchange tests.

## 3. Target Layout

```text
winnow/
├── src/
│   └── winnow/
│       ├── __init__.py
│       ├── collect.py
│       ├── question.py
│       ├── state.py
│       ├── results.py
│       ├── stopping.py
│       ├── config.py
│       ├── exceptions.py
│       ├── parser/
│       ├── estimator/
│       │   └── statistics.py
│       └── exchange/
│           ├── __init__.py
│           ├── client.py
│           └── logging.py
└── tests/
    ├── parser/
    ├── estimator/
    └── exchange/
```

## 4. Migration Sequence

The migration is divided into five independently reviewable phases:

1. `01-public-contract-and-baseline.md` fixes the intended facade and protects existing behaviour.
2. `02-estimator-statistics.md` moves shared statistical mathematics to its owner.
3. `03-exchange-boundary.md` establishes the external query and diagnostic I/O boundary.
4. `04-imports-and-test-ownership.md` enforces dependency direction and aligns test placement.
5. `05-documentation-and-validation.md` completes repository hygiene and downstream verification.

Each phase should leave the test suite passing. Structural moves should not be combined with unrelated behavioural or syntax modernisation.

## 5. Baseline

At the time of planning, all 142 tests pass. Mypy and Pyright report no type errors. Ruff reports 16 existing findings, principally import ordering and Python 3.13 generic syntax recommendations.

The existing Ruff findings are not evidence of migration regressions. Import findings in touched files may be resolved naturally, but unrelated generic-syntax changes should remain a separate task.

## 6. Completion

The migration is complete when the target ownership boundaries are visible in the directory tree, internal imports follow the dependency rules, only the intended facade is public, the source and test trees agree, and the Winnow and relevant Dietrix checks pass.
