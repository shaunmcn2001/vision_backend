# Collaboration Guidelines for Vision Backend

Welcome! This document captures the conventions and workflow we expect every contributor—including future AI agents—to follow when working in this repository.

## Coding Conventions
- **Language style**: Match the existing Python and FastAPI patterns. Prefer explicit imports, type hints for public functions, and docstrings describing request/response models.
- **Formatting**: Run `ruff format` (once configured) or adhere to Black-compatible formatting—4 space indentation, double quotes, and trailing commas where they aid diffs.
- **Naming**: Use descriptive, lowercase module names and `snake_case` for variables/functions. Reserve `CamelCase` for Pydantic models and classes.
- **Error handling**: Surface informative HTTP exceptions using FastAPI's `HTTPException`. Log unexpected exceptions before bubbling them up.

## Required Checks
Before opening a PR or finalizing work, complete the following (or document why a step was skipped if tooling is unavailable):
1. If a `pyproject.toml` is present, run `poetry install` once to sync dependencies.
2. Lint: `poetry run ruff check .` (or `ruff check .` when Poetry is not used).
3. Format verification: `poetry run ruff format --check .` (or `ruff format --check .`).
4. Tests: `poetry run pytest` (or `pytest`).
5. Type checking for API contract changes: `poetry run mypy` (or `mypy`).
Always record skipped or failing checks with context in your final summary.

## API Testing Expectations
- Exercise every modified endpoint using automated tests (`pytest`) or the `/docs` Swagger UI when adding new routes.
- Provide sample requests and responses in documentation updates when behavior changes.
- For breaking changes, add regression tests that demonstrate the new contract.

## Geospatial Data Handling Nuances
- Always confirm coordinate reference systems (CRS). Unless explicitly required otherwise, default to `EPSG:4326` for inputs and `EPSG:3857` for Earth Engine sampling to maintain metre-scale fidelity.
- Validate incoming GeoJSON geometries using Shapely or Pydantic validators; reject invalid or mixed-CRS payloads.
- Document CRS assumptions and transformations in code comments and user-facing docs to avoid ambiguity.

## Documenting Completed Work
- Update relevant Markdown files (e.g., `README.md`, endpoint docs) whenever behavior, dependencies, or workflows change.
- Each commit message should include a **Notes** section summarizing the change and its impact.
- Maintain changelog-style bullets in the repository README under an "Updates" section (create it if absent) after each significant modification.

## Recording Updates After Each Change
After every change set:
1. Append a dated entry under the README "Updates" section describing the change, tests run, and any follow-up actions.
2. If no README section exists, add one with reverse-chronological entries.
3. Reference relevant issues or tickets when available.

Following these guidelines keeps our collaboration consistent, auditable, and helpful for the next contributor.
