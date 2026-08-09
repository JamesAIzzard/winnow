# Phase 03: Completed Exchange Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve each full prompt-response interaction across the exchange boundary so collection can report it later.

**Architecture:** `ExchangeRecordingClient` remains responsible for building the prompt, awaiting the caller, and recording the exchange. It returns a `QuestionInteraction`; `collect()` consumes its raw response while temporarily retaining the old callback signature.

**Tech Stack:** Python 3.13 async protocols and dataclasses, pytest

## Global Constraints

Preserve build prompt, await query, record exchange, then parse response. Do not change JSONL fields, callback behaviour or wave concurrency in this phase. End the phase with passing tests.

---

Status: Draft for review

Depends on: Phase 02

### Task 1: Return the completed interaction

**Files:**

- Modify: `src/winnow/exchange/client.py`
- Modify: `tests/exchange/test_client.py`

**Interfaces:**

- Consumes: `QuestionInteraction` from `winnow.progress`
- Produces: `ExchangeRecordingClient.query_prompt(prompt: Prompt[T]) -> QuestionInteraction`

- [ ] **Step 1: Change the exchange test expectation**

Replace its response assertion with:

```python
interaction = asyncio.run(client.query_prompt(prompt))

prompt_body = prompt.build_prompt()
assert interaction == QuestionInteraction(
    question_uid="protein",
    prompt=prompt_body,
    raw_response=f"response to {prompt_body}",
)
assert events[-1] == (
    "record",
    {
        "uid": interaction.question_uid,
        "prompt": interaction.prompt,
        "response": interaction.raw_response,
    },
)
```

Import `QuestionInteraction` from `winnow` in this focused boundary test.

- [ ] **Step 2: Confirm that the old return type fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/exchange/test_client.py -q`

Expected: the returned `str` does not equal `QuestionInteraction`.

- [ ] **Step 3: Construct the interaction at the exchange boundary**

Change `query_prompt` to:

```python
async def query_prompt(self, prompt: Prompt[T]) -> QuestionInteraction:
    prompt_body = prompt.build_prompt()
    raw_response = await self.query_fn(prompt_body)
    interaction = QuestionInteraction(
        question_uid=prompt.uid,
        prompt=prompt_body,
        raw_response=raw_response,
    )
    record_exchange(
        uid=interaction.question_uid,
        prompt=interaction.prompt,
        response=interaction.raw_response,
    )
    return interaction
```

Import the shared internal concept using `from ..progress import QuestionInteraction`.

### Task 2: Consume interactions without changing progress callbacks

**Files:**

- Modify: `src/winnow/collect.py`
- Test: `tests/test_collect.py`

**Interfaces:**

- Consumes: `tuple[QuestionInteraction, ...]` returned by the exchange wrapper
- Preserves: the temporary two-argument callback until Phase 04

- [ ] **Step 1: Adapt collection to the new return value**

Rename `responses` to `interactions` and process each value with:

```python
for question, interaction in zip(wave, interactions):
    _process_response(question, interaction.raw_response, states)
```

Do not emit `CollectionProgress` yet.

- [ ] **Step 2: Verify the exchange and collection boundaries**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/exchange/test_client.py tests/exchange/test_logging.py tests/test_collect.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: the focused checks and complete suite pass, including concurrency and the old progress tests.

- [ ] **Step 3: Commit the phase**

```powershell
git add src/winnow/exchange/client.py src/winnow/collect.py tests/exchange/test_client.py
git commit -m "Carry completed query interactions"
```

## Exit Criteria

The exchange wrapper returns exactly what was queried and received, diagnostic logging is unchanged, and collection results remain unchanged.
