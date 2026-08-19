# Contributing

This repository is edited entirely through the GitHub web UI (Add file →
Create/Upload, the pencil/edit button, and the trash icon), there is no
local git workflow assumed here. Keep that in mind when proposing changes:
PRs should be small, self-contained, and reviewable file-by-file.

## Before opening a PR

1. Run the quality gate locally if you have a full dev environment
   (`pip install -r requirements-dev.txt`, or `pip install -e .[dev]` after
   this PR's packaging change):
   ```bash
   make check   # ruff, black --check, mypy, pytest
   ```
   or individually: `ruff check .`, `black --check .`, `mypy .`, `pytest -q`.
2. If you touch `src/value_decomposition.py`, `src/data_prep.py`,
   `src/features.py`, or `src/train_model.py`, verify behavior is unchanged
   for any refactor: recompute outputs before and after your change and
   confirm they match (see recent PRs for the pattern, running the
   pipeline twice and diffing the JSON output).
3. Keep changes minimal and scoped to one concern per PR. Don't rename or
   move files unless the PR is specifically about that.
4. Match the existing style: `from __future__ import annotations` at the
   top of every module, full type annotations on all function signatures
   (`mypy`'s `disallow_untyped_defs` is enabled), and tests that use the
   isolated `tmp_path`-based fixtures in `tests/conftest.py` rather than
   touching real project data.

## Reporting issues

Open a GitHub issue describing the problem, the file/function involved if
known, and for bugs, the exact command or code path that reproduces it.
