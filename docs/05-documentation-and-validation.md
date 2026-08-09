# Phase 05: Documentation and Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release the breaking callback contract as version `0.2.0`, synchronise user-facing documentation with the executable API and validate the finished package.

**Architecture:** The README remains a concise usage guide, while the Lodestone Winnow notes remain the complete API and structural authority. Validation exercises the top-level facade rather than internal import paths.

**Tech Stack:** GFM, Python 3.13, pytest, Ruff, mypy, Pyright, Hatchling

## Global Constraints

Use British English and clean Git-tracked Markdown. Set the project version to exactly `0.2.0`. Keep JSONL logging optional, keep historical progress persistence with the caller, and document `CollectionProgress`, `QuestionInteraction`, `QuestionUID`, `SampleStates` and `ProgressCallback` as the public progress vocabulary. End the phase with the full repository cleanly verified.

---

Status: Draft for review

Depends on: Phase 04

### Task 1: Bump the package version

**Files:**

- Modify: `pyproject.toml`

**Interfaces:**

- Produces: project version `0.2.0`

- [ ] **Step 1: Change the project version**

Set the project metadata to:

```toml
version = "0.2.0"
```

- [ ] **Step 2: Verify the declared version**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import pathlib, tomllib; project = tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8'))['project']; assert project['version'] == '0.2.0'"
```

Expected: exit code `0`.

### Task 2: Replace obsolete README examples

**Files:**

- Modify: `README.md`

**Interfaces:**

- Documents: `Prompt`, `Question`, `CollectionProgress`, `NoEstimate`, `Estimate`, `NeedsReview`, `collect`
- Removes: direct `Question(uid=..., query=..., parser=...)` construction and the two-argument progress callback

- [ ] **Step 1: Update simple usage**

Construct questions through `Prompt`:

```python
Question(
    prompt=Prompt(
        uid="protein",
        query="How many grams of protein are in 100g of chicken breast?",
        parser=FloatParser(),
    ),
    estimator=NumericalEstimator(),
    stopping_criterion=StoppingCriterion(),
)
```

Handle `Estimate | NeedsReview` explicitly rather than assuming every result has `value` and `confidence`.

- [ ] **Step 2: Update progress usage**

Use the public envelope directly:

```python
def show_progress(progress: CollectionProgress) -> None:
    for interaction in progress.new_interactions:
        print(f"{interaction.question_uid}: {interaction.raw_response}")
    for uid, state in progress.sample_states.items():
        if state.current_estimate is not NoEstimate:
            print(f"{uid}: {state.current_estimate} ({state.current_confidence:.0%})")
```

State plainly that Winnow does not retain progress history for the caller.

- [ ] **Step 3: Repair visible encoding damage**

Replace mojibake such as `â€”` with normal British-English punctuation. Do not introduce em dashes.

### Task 3: Synchronise the complete design notes

**Files:**

- Revise through Lodestone: `Winnow Docs.md`
- Revise through Lodestone: `Winnow Project Structure.md`

**Interfaces:**

- Documents: the complete public progress vocabulary
- Documents: `progress.py` ownership and the `exchange -> progress` dependency

- [ ] **Step 1: Confirm the public progress vocabulary in the API note**

Retain `QuestionUID`, `SampleStates` and `ProgressCallback` in the documented public signatures alongside the two concrete dataclasses. Explain that these names describe the supported collection boundary and that callers own any historical copies.

- [ ] **Step 2: Update the project-structure note**

Add `progress.py` to the root layout and responsibility table. Permit `exchange` to depend on `progress` for `QuestionInteraction`; retain the prohibition on depending on `collect`, parsers or estimators.

Use inline green review marks because these Lodestone notes are outside Git.

### Task 4: Validate the completed migration

**Files:**

- Verify: all changed source, tests and documentation

**Interfaces:**

- Verifies: installed top-level imports and documented callback behaviour

- [ ] **Step 1: Run the complete automated checks**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\pyright.exe
```

Expected: pytest, mypy and Pyright pass; Ruff introduces no new findings. Any repository baseline findings must be reported separately from migration regressions.

- [ ] **Step 2: Smoke-test the documented facade**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from winnow import CollectionProgress, ProgressCallback, QuestionInteraction, QuestionUID, SampleStates, Prompt, Question, collect"
```

Expected: exit code `0`.

- [ ] **Step 3: Review the diff and repository state**

Confirm that no compatibility shim, provider-specific code or client-side history mechanism was introduced. Check that every new source file has an evident test owner.

- [ ] **Step 4: Commit the phase**

```powershell
git add pyproject.toml README.md
git commit -m "Release progress API version"
```

Commit the Lodestone note changes separately through their normal documentation workflow rather than adding them to this Git repository.

## Exit Criteria

The package declares version `0.2.0`, the README examples match the installed facade, the complete notes match the implementation and ownership rules, and all relevant automated checks pass.
