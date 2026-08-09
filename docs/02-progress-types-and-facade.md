# Phase 02: Progress Types and Facade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce the documented progress-reporting types without changing collection behaviour.

**Architecture:** `question.py` owns the shared `QuestionUID` alias. A focused root-level `progress.py` owns `CollectionProgress`, `QuestionInteraction`, `SampleStates` and `ProgressCallback`. The package root re-exports all five names as part of Winnow's supported facade.

**Tech Stack:** Python 3.13 dataclasses and typing, pytest

## Global Constraints

Export the complete documented progress vocabulary. Order the principal progress value before its supporting interaction and alias definitions. End the phase with passing tests.

---

Status: Draft for review

Depends on: Phase 01

### Task 1: Define and expose progress values

**Files:**

- Create: `src/winnow/progress.py`
- Modify: `src/winnow/__init__.py`
- Modify: `src/winnow/question.py`

**Interfaces:**

- Produces: public `type QuestionUID = str`
- Produces: `CollectionProgress(sample_states: SampleStates, new_interactions: tuple[QuestionInteraction, ...])`
- Produces: `QuestionInteraction(question_uid: QuestionUID, prompt: str, raw_response: str)`
- Produces: `type SampleStates = Mapping[QuestionUID, SampleState[Any]]`
- Produces: `type ProgressCallback = Callable[[CollectionProgress], None]`
- Produces: top-level imports for all five progress-reporting types

- [ ] **Step 1: Add the question UID alias**

After the imports and declarations that frame `question.py`, define:

```python
type QuestionUID = str
```

Use `QuestionUID` for question identifiers, question-bank keys and related method annotations within `question.py`.

- [ ] **Step 2: Add the progress module**

Create `src/winnow/progress.py` with this public shape:

```python
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .question import QuestionUID
from .state import SampleState


@dataclass(frozen=True, kw_only=True)
class CollectionProgress:
    sample_states: SampleStates
    new_interactions: tuple[QuestionInteraction, ...]


@dataclass(frozen=True, kw_only=True)
class QuestionInteraction:
    question_uid: QuestionUID
    prompt: str
    raw_response: str


type SampleStates = Mapping[QuestionUID, SampleState[Any]]
type ProgressCallback = Callable[[CollectionProgress], None]
```

- [ ] **Step 3: Extend the package facade**

Import `QuestionUID` from `question.py`. Import `CollectionProgress`, `QuestionInteraction`, `SampleStates` and `ProgressCallback` from `progress.py`. Insert all five names into the alphabetised `__all__`.

- [ ] **Step 4: Verify the facade and current package**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from winnow import CollectionProgress, ProgressCallback, QuestionInteraction, QuestionUID, SampleStates"
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: the public imports succeed and the complete suite passes. The collection callback remains unchanged in this phase.

- [ ] **Step 5: Commit the phase**

```powershell
git add src/winnow/progress.py src/winnow/question.py src/winnow/__init__.py
git commit -m "Add public progress value types"
```

## Exit Criteria

All five progress-reporting types are usable from `winnow`, internal annotations use the same vocabulary, and no runtime collection behaviour has changed.
