set shell := ["bash", "-cu"]

# Default — list recipes.
default:
    @just --list

# Install / refresh dev dependencies.
sync:
    uv sync

# Full quality gate — lint, format-check, type-check, tests.
check: lint format-check typecheck test

# Lint only.
lint:
    uv run ruff check src tests

# Format-check (no rewrite).
format-check:
    uv run ruff format --check src tests

# Type-check with mypy strict.
typecheck:
    uv run mypy

# Run tests.
test:
    uv run pytest

# Auto-fix ruff lint + apply formatting.
fix:
    uv run ruff check --fix src tests
    uv run ruff format src tests
