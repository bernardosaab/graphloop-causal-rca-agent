.PHONY: install lint format typecheck test check bench demo

install:
	uv sync

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck:
	uv run mypy src

test:
	uv run pytest --cov

check: lint typecheck test

bench:
	uv run graphloop bench --scenarios 100 --seed 7 --out results

demo:
	uv run graphloop run --incident 3 --agent graph+memory --approve prompt
