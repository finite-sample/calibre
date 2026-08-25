.PHONY: help install dev test lint format clean docs build

help:
	@echo "Available commands:"
	@echo "  install    Install package in production mode"
	@echo "  dev        Install all dependency groups and the pre-commit hook"
	@echo "  test       Run tests with coverage"
	@echo "  lint       Run the checks CI's lint job runs"
	@echo "  format     Format code with ruff"
	@echo "  clean      Remove build artifacts and cache files"
	@echo "  docs       Build documentation (warnings are errors, as in CI)"
	@echo "  build      Build distribution packages"

install:
	uv pip install .

dev:
	uv sync --all-groups
	uv run pre-commit install

test:
	uv run pytest tests/ -v

# Mirrors py-canon's reusable-ci lint job: ruff, pydoclint (pinned to the same
# version CI uses), pyright.
lint:
	uv run ruff check .
	uv run ruff format --check .
	uvx --from pydoclint==0.9.1 pydoclint calibre
	uv run pyright

format:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete

docs:
	uv run sphinx-build -W -b html docs docs/_build/html
	@echo "Documentation built at docs/_build/html/index.html"

build: clean
	uv build
