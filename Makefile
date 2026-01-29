.PHONY: fmt lint typecheck test check

fmt:
	poetry run ruff format .

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy .

test:
	poetry run pytest

check: fmt lint typecheck

ruff: fmt lint
