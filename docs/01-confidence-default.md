# Phase 01: Confidence Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the executable default confidence threshold match the documented value of `0.90`.

**Architecture:** `WinnowConfig` remains the single owner of package defaults. `StoppingCriterion` continues to derive its default from `default_config`.

**Tech Stack:** Python 3.13, pytest

## Global Constraints

Use British English and keep Git-tracked Markdown clean. Preserve the existing stopping order and change no explicitly supplied threshold. End the phase with passing tests.

---

Status: Draft for review

Depends on: None

### Task 1: Align the confidence default

**Files:**

- Modify: `src/winnow/config.py`

**Interfaces:**

- Consumes: `default_config.standard_confidence`
- Produces: `StoppingCriterion().confidence_threshold == 0.90`

- [ ] **Step 1: Change the owned default**

In `WinnowConfig`, set:

```python
standard_confidence: float = 0.90
```

Do not duplicate `0.90` in `StoppingCriterion`.

- [ ] **Step 2: Verify the existing suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_stopping.py tests/test_state.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: the focused stopping checks and complete suite pass.

- [ ] **Step 3: Commit the phase**

```powershell
git add src/winnow/config.py
git commit -m "Align default confidence threshold"
```

## Exit Criteria

The default is owned in one place, the public stopping criterion exposes `0.90`, and explicitly configured thresholds behave unchanged.
