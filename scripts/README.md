# Scripts Overview

This folder contains helper scripts for the `langextract-azureopenai` plugin.

## bump_version.py
- Purpose: Bump the package version using SemVer (patch/minor/major).
- Actions: Reads current version from `langextract_azureopenai/__init__.py`, computes next version, updates both `__init__.py` and `pyproject.toml`.
- Usage: `python scripts/bump_version.py [patch|minor|major]`

## run_tests.py
- Purpose: One-command, comprehensive local test runner.
- Actions:
  - Installs dev deps via `uv sync --extra dev` if available; otherwise `pip install -e .[dev]`.
  - Runs formatting checks (black, isort), lint (ruff), type-check (mypy), and unit tests (`pytest -m unit`).
  - If `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and `AZURE_OPENAI_API_VERSION` are set, runs a targeted integration test.
  - Generates coverage (terminal + HTML) and validates the build via `scripts/check_build.py`.
- Usage: `python scripts/run_tests.py`

## check_build.py
- Purpose: Validate that the package builds, installs, and imports cleanly.
- Actions:
  - Requires `uv`; cleans `dist/`, `build/`, `*.egg-info/`, then runs `uv build`.
  - Lists build artifacts and wheel contents.
  - Creates a temp venv with `uv venv`, installs the wheel via `uv pip`, and verifies import + `__version__`.
- Usage: `python scripts/check_build.py`

## release.py
- Purpose: Interactive release helper for the plugin.
- Actions:
  - Checks git status; runs `scripts/run_tests.py` (continues on failure only with confirmation).
  - Prompts for version bump and runs `scripts/bump_version.py`.
  - Builds with `uv build`, validates with `scripts/check_build.py`.
  - Creates a git tag `vX.Y.Z`, optionally pushes commits/tags, and can publish to PyPI via `uv publish --token`.
- Usage: `python scripts/release.py`

## Notes
- `uv` is preferred by several scripts. If you don’t have it installed, `run_tests.py` falls back to standard tools; `check_build.py` and `release.py` currently expect `uv` to be available.
