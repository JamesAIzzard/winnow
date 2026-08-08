# Phase 03: Exchange Boundary

Status: Draft for review

Depends on: Phase 02

## 1. Objective

Create the `exchange` package as the narrow boundary through which prompts and responses cross between Winnow and a caller-supplied source.

## 2. Ownership

`exchange/client.py` owns the public query protocol and the private wrapper used by collection to build prompts, invoke the supplied query function and record completed exchanges.

`exchange/logging.py` owns JSONL logger configuration and private exchange-record emission. Diagnostic logging remains optional and source-neutral.

Provider-specific clients do not belong in Winnow. OpenAI, Anthropic or other adapters, credentials, rate-limit handling and retry policies remain caller responsibilities.

## 3. Planned Changes

Create:

```text
src/winnow/exchange/
├── __init__.py
├── client.py
└── logging.py
```

Move `LLMClient` from `llm_client.py` to `exchange/client.py` without changing its callable contract. Rename `LoggedLLMClient` as a private implementation, using an intention-revealing name that describes query execution with exchange recording.

Move `configure_jsonl_logging` from `jsonl_logging.py` to `exchange/logging.py`. Move `log_exchange` alongside it and rename it as a private function.

Update `collect.py` to use the exchange protocol and private wrapper. Preserve the order of operations: build the prompt, await the caller-supplied query, record the raw exchange, then parse the response.

Update the top-level facade so `LLMClient` and `configure_jsonl_logging` retain their documented imports from `winnow`. Do not expose private exchange implementations through `exchange/__init__.py` or the package root.

Delete `llm_client.py` and `jsonl_logging.py` after all references have moved.

## 4. Tests

Move the current query-wrapper test out of `tests/test_question.py` into `tests/exchange/test_client.py`. Add `tests/exchange/test_logging.py` for JSONL record structure, UTC timestamps, handler replacement and silent operation when no file handler is configured.

Keep collection-level tests focused on observable orchestration: supplied queries are awaited, waves remain concurrent, progress is reported after processing, and final results are unchanged.

## 5. Dependency Rules

`exchange` may depend on `question` and the Python standard library. It must not depend on `collect`, parser or estimator implementations.

`collect` may depend on the exchange contract and private exchange wrapper. External callers depend only on the top-level facade.

## 6. Out of Scope

This phase does not change prompts, progress payloads, logging format, concurrency, source retries or provider integration.

## 7. Verification

Run the exchange, question and collection tests, followed by the complete suite and type checks. Search for the removed module names and public implementation names. Confirm that documented top-level imports still work.

## 8. Exit Criteria

All external query and diagnostic I/O code lives under `exchange`. Only `LLMClient` and `configure_jsonl_logging` cross the public facade. Collection behaviour and JSONL output are unchanged.
