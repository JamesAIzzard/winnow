# Phase 05: Documentation and Validation

Status: Draft for review

Depends on: Phase 04

## 1. Objective

Complete the migration by synchronising repository documentation, checking package contents and verifying the principal downstream consumer.

## 2. Repository Documentation

Update `README.md` to match the public API described in `Winnow Docs`. Its examples should use `Prompt`, the current progress callback contract, `Estimate | NeedsReview`, and imports from the top-level facade.

Document the source-agnostic query boundary without presenting a provider client as part of Winnow. Keep JSONL logging as optional diagnostics rather than the caller's primary integration seam.

Do not duplicate the full Lodestone design notes in the repository. The README should explain how to use the released library, while these migration documents record the temporary implementation sequence.

## 3. Package Hygiene

Build the distribution and inspect its contents. Confirm that `src/winnow/py.typed` is included and determine whether the duplicate repository-level `py.typed` serves any packaging purpose. Remove it if it does not.

Confirm that obsolete `util.py`, `llm_client.py` and `jsonl_logging.py` modules are absent from the built package. Import the documented facade from the built distribution in a clean environment.

## 5. Full Verification

Run:

- the complete Winnow test suite;
- Ruff, comparing any remaining findings with the recorded baseline;
- mypy and Pyright;
- a clean package build and facade import smoke test;

Review the final source and test trees against `00-migration-overview.md`. Check that no compatibility shim or temporary migration file remains unintentionally.

## 6. Exit Criteria

The repository matches the target structure, the README matches the executable API, the built distribution contains the intended files, and both Winnow and its principal downstream consumer validate successfully.
