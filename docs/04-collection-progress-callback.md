# Phase 04: Collection Progress Callback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old two-argument callback with the documented single `CollectionProgress` value.

**Architecture:** `collect()` gathers a wave of `QuestionInteraction` values, processes every raw response, then emits one progress envelope containing the current cumulative states. Persistence remains a caller responsibility.

**Tech Stack:** Python 3.13 asyncio, pytest, mypy, Pyright

## Global Constraints

Call progress once after each dispatched wave and never before collection or as a separate completion event. Include declines and parse failures in `new_interactions`. Do not copy state for historical retention. End the phase with passing tests and type checks.

---

Status: Draft for review

Depends on: Phase 03

### Task 1: Specify the new callback behaviour

**Files:**

- Modify: `tests/test_collect.py`

**Interfaces:**

- Consumes: `CollectionProgress`
- Replaces: `Callable[[dict[str, SampleState], frozenset[str]], None]`
- Produces: `ProgressCallback`

- [ ] **Step 1: Convert existing callback tests to the new value**

Import `CollectionProgress` and replace two-argument callbacks with either `progress_events.append` or:

```python
def on_progress(progress: CollectionProgress) -> None:
    progress_events.append(progress)
```

Read state through `progress.sample_states[question_uid]`. Replace wave-UID assertions with interaction UID assertions.

- [ ] **Step 2: Add one focused wave-payload test**

For a two-question wave returning one valid response and one unparseable response, assert:

```python
assert tuple(
    interaction.question_uid
    for interaction in progress.new_interactions
) == ("protein", "fat")
assert progress.new_interactions[0].raw_response == "31"
assert progress.new_interactions[1].raw_response == "not a number"
assert progress.sample_states["protein"].query_count == 1
assert progress.sample_states["fat"].parse_failure_count == 1
```

Also assert that each interaction prompt contains its original query and the `DECLINE` instruction.

- [ ] **Step 3: Confirm that the old implementation fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_collect.py -q`

Expected: callbacks fail because `collect()` still supplies two positional arguments.

### Task 2: Emit `CollectionProgress`

**Files:**

- Modify: `src/winnow/collect.py`

**Interfaces:**

- Consumes: `CollectionProgress`, `QuestionInteraction`, cumulative `states`
- Produces: `on_progress: ProgressCallback | None`
- Produces: `initial_states: SampleStates | None`

- [ ] **Step 1: Update collection annotations and imports**

Import `QuestionUID` from `.question`, and `CollectionProgress`, `ProgressCallback` and `SampleStates` from `.progress`. Remove the now-obsolete `Callable` and `frozenset` callback concept.

Use the public aliases in the collection signature:

```python
on_progress: ProgressCallback | None = None,
initial_states: SampleStates | None = None,
```

Use `SampleStates` for the corresponding inputs to `_validate_initial_state_uids` and `_initialise_states`. Use `QuestionUID` for question-keyed result and internal mapping annotations.

- [ ] **Step 2: Emit one envelope after processing the wave**

Replace the old callback invocation with:

```python
if on_progress is not None:
    on_progress(CollectionProgress(
        sample_states=states,
        new_interactions=tuple(interactions),
    ))
```

Do not copy `states`; callers own any historical persistence.

- [ ] **Step 3: Verify callback semantics**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_collect.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: the focused checks and complete suite pass. Callbacks occur once per dispatched wave, after response processing, and receive complete interactions plus cumulative states.

- [ ] **Step 4: Verify static types**

Run:

```powershell
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\pyright.exe
```

Expected: both checks complete without errors.

- [ ] **Step 5: Commit the phase**

```powershell
git add src/winnow/collect.py tests/test_collect.py
git commit -m "Report collection progress by wave"
```

## Exit Criteria

The documented callback shape is executable, every completed exchange is reported once, and clients remain responsible for historical persistence.
