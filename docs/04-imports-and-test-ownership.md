# Phase 04: Imports and Test Ownership

Status: Draft for review

Depends on: Phase 03

## 1. Objective

Make the target dependency direction visible in imports and align test placement with source ownership.

## 2. Internal Imports

Convert imports inside `src/winnow` to relative imports from the module that owns each concept. Internal code must not import through the top-level `winnow` facade.

Apply these dependency rules:

| Code area | May depend on |
| --- | --- |
| Root modules | Other root modules and subpackage contracts as required |
| `parser` | `config`, `exceptions` and parser-private modules |
| `estimator` | `state` and estimator-private modules |
| `exchange` | `question` and the Python standard library |
| Top-level `__init__.py` | Static names intentionally exposed to callers |

`parser`, `estimator` and `exchange` must not depend on `collect`. Parser and estimator implementations must not depend on each other.

## 3. Test Imports

Facade tests and collection tests should import public concepts from `winnow`, matching real client usage. Focused parser, estimator and exchange tests may import their internal implementation modules where direct edge-case coverage is clearer.

Move tests only when their responsibility is currently misplaced. Do not create empty test modules merely to reproduce the illustrative target tree.

Keep shared fixtures in `tests/conftest.py` only when several test areas use them. Local fixtures should remain beside the tests they support.

## 4. Out of Scope

This phase does not split central root modules, introduce interface-only packages, or make private subpackages into additional public APIs.

## 5. Verification

Run the complete test suite and type checks. Search source imports to confirm that internal absolute `winnow.*` imports have been removed except from the facade where appropriate, and inspect the remaining imports against the dependency table. Exercise the installed package rather than relying on repository-root import accidents.

## 6. Exit Criteria

Imports communicate ownership consistently, no forbidden dependency direction remains, and each test has an evident source or facade owner.
