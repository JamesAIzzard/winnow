## Run commands
- You are running on windows, please use powershell commands.
- Tests: `poetry run pytest -q`
- Lint: `poetry run ruff check .`
- Lint fix: `poetry run ruff check . --fix`
- Type check: `poetry run mypy .`
- `rg` (ripgrep) is installed and is available on PATH.

## Quick diagnostics
- Venv path: `poetry env info --path`
- Python used: `poetry run python -c "import sys; print(sys.executable)"`
- Tool path: `poetry run which ruff` / `poetry run which pytest`

## Coding Guidelines
- Stick to just the task at hand. Avoid changing other parts of the code which are
  not directly related to the task.
- Follow PEP 8 via `ruff --fix`.
- All imports should be at the top of the file, and ordered as follows:
  1. Standard library imports
  2. Third-party imports
  3. Local application imports
  4. if TYPE_CHECKING, imports for type hints only
- Always use British English spelling for code and comments.
- Since we use from __future__ import annotations, type hints generally should not be quoted. Use TYPE_CHECKING to import types that are only needed for type hints.
- Self documenting function names are better than comments.
- Prefer small, verb-noun functions (≈ 5–20 lines).
- Respecting SRP is vital.
- Prefer keyword only arguments for public methods, unless there is only a single obvious argument.
- Guard clauses beat deep nesting. Replace pyramids of if … else with early returns and continue statements.
- Code must be DRY, but not to the point where it is obfuscated.
- Side-effect–free functions are preferred. When possible make them pure (no I/O, no globals).
- Create encapsulation, i.e account.debit(£5) is better than account.balance -= 5.
- Separate command from query. Functions that change state shouldn’t also return useful data; returning None makes intent explicit.
- Implement dependency inversion via dependency injection.
- Favour composition over inheritance.
- Readable is better than clever. Always favour easy to understand code over clever tricks.
- Docstrings and comments should explain why. Code should explain what. Code should be self-documenting, comments used sparingly.
- Whenever you touch a file, prune dead code, rename unclear vars, add a missing test.
- Use ruff and mypy to check code style and type hints.

## Testing Guidelines
- Run tests
  - python -m pytest -q

- Location & naming
  - Keep everything in the root-level tests/ directory.
  - Every file starts with test\_.
    - Single-file class example: tests/test\_foo\_service.py.
    - Large class: create a folder (e.g. tests/foo\_service/) and split by behaviour (test\_foo\_service\_behaviour\_error.py, etc.).

- Structure inside a file
  - Group related checks in a class, e.g. class TestErrorDetection:.
  - Every test method begins with test\_.
  - If a file grows too long, split it—never let one file balloon.

- Docstring rule
  - Exactly one docstring per test method.
  - Must start with “Verify …”
    - """Verify raises exception when widget is not found."""

- Fixtures
  - Live in tests/fixtures/.
  - Loaded automatically as pytest plug-ins—inject them by name.
  - Always type-hint injected fixtures; wrap the import in TYPE\_CHECKING if it isn’t needed at runtime.

- Good examples
  - See tests/components\_tests/model/ for reference-quality patterns.

