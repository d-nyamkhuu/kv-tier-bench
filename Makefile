# Everything here runs without a GPU.
.PHONY: lint test

PY_SRC := src tests

lint:
	uv run ruff check $(PY_SRC) && uv run ruff format --check $(PY_SRC)

test:
	uv run pytest
