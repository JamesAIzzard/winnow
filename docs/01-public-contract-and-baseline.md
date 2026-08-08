# Phase 01: Public Contract and Baseline

Status: Draft for review

## 1. Objective

Define the public contract before moving implementation files. This gives later phases a stable seam and distinguishes intended compatibility from accidental exposure.

## 2. Public Facade

`winnow/__init__.py` remains the sole client-facing facade. Its `__all__` declaration is the authoritative list of supported imports.

The facade retains its current public types, exceptions, parsers, estimators, stopping policy, `collect`, `LLMClient` and `configure_jsonl_logging`. The contract for this migration is the current `__all__` declaration minus the two agreed removals below.

`LoggedLLMClient` and `log_exchange` are removed from `__all__` and from the names imported into the package root. They are source-neutral implementation details used to add diagnostics around the caller-supplied query function.

No deprecated compatibility exports are planned. Winnow is at version `0.1.0`, and the known Winnow and Dietrix consumers do not import either name.

## 3. Planned Changes

Add a facade contract test that imports the documented API from `winnow` and checks `winnow.__all__` explicitly. The test should fail if a public name disappears or a private implementation name is added unintentionally.

Retain the existing behavioural tests as the migration baseline. Record the existing Ruff findings so later phases distinguish pre-existing style debt from new failures.

## 4. Out of Scope

This phase does not move modules, rename `LLMClient`, change collection behaviour, add public types proposed elsewhere in `Winnow Docs`, rewrite downstream imports, or modernise generic type syntax.

## 5. Verification

Run the complete test suite, Ruff, mypy and Pyright. Confirm that the facade test passes and that the reported Ruff delta contains no new findings.

## 6. Exit Criteria

The intended public names are executable and type-checkable from `winnow`. `LoggedLLMClient` and `log_exchange` are no longer public. All existing behaviour remains unchanged.
