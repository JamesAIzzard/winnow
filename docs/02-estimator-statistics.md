# Phase 02: Estimator Statistics

Status: Draft for review

Depends on: Phase 01

## 1. Objective

Move shared statistical mathematics from the generic root-level `util.py` module to the estimator package that owns it.

## 2. Ownership

`median` and `mad` change because estimators require different statistical judgements. Their owner is therefore `estimator`, not a package-wide utility module.

The target module is `src/winnow/estimator/statistics.py`. It remains private implementation code and is not exported through either `estimator/__init__.py` or the top-level facade.

## 3. Planned Changes

Move the contents of `src/winnow/util.py` to `src/winnow/estimator/statistics.py` without behavioural changes.

Update `numerical.py` and `optional_int.py` to import `median` and `mad` relatively from `.statistics`. Delete `util.py` once no references remain.

Keep estimator behaviour covered by the existing numerical and optional-integer tests. Add direct tests for `statistics.py` only where a mathematical edge case cannot be expressed clearly through an estimator's public behaviour.

## 4. Dependency Rule

`estimator/statistics.py` may use only the Python standard library. It must not depend on parser, collection, exchange or package-facade modules.

## 5. Out of Scope

This phase does not change formulas, confidence semantics, rounding, zero handling, estimator protocols or public exports. It does not create a replacement generic utility package.

## 6. Verification

Run the estimator tests and complete test suite. Search the repository for `winnow.util`, `from .util`, `median` and `mad` to confirm that every remaining reference has the intended owner. Run mypy and Pyright.

## 7. Exit Criteria

`util.py` no longer exists. Estimator outputs are unchanged. Shared statistical code has one explicit owner and introduces no new dependency direction.
