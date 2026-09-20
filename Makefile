.PHONY: help install dev test lint format clean docs build ci-docker paper paper-check

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
	@echo "  paper      Generate manuscript exhibits and compile the PDF"
	@echo "  paper-check  Execute the manuscript example and artifact tests"
	@echo "  granularity-run  Fit and evaluate the granularity simulation"
	@echo "  granularity  Build its notebook, figures, and PDF note"
	@echo "  granularity-check  Test decision identities and artifacts"
	@echo "  ci-docker  Run the release checks in Python 3.12 on Linux"

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

paper:
	uv run python -m benchmarks.paper
	latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -Werror -outdir=build ms/calibre.tex

paper-check:
	uv run python ms/example.py
	uv run pytest tests/test_paper.py tests/test_benchmarks.py

ci-docker:
	docker run --rm --pull=always \
		--mount type=bind,source="$(CURDIR)",target=/workspace,readonly \
		--workdir /workspace \
		--env UV_PROJECT_ENVIRONMENT=/tmp/calibre-venv \
		--env UV_CACHE_DIR=/tmp/uv-cache \
		--env RUFF_CACHE_DIR=/tmp/ruff-cache \
		--env COVERAGE_FILE=/tmp/.coverage \
		ghcr.io/astral-sh/uv:0.12.5-python3.12-trixie \
		sh -c 'uv sync --locked --all-groups && \
			uv run pytest tests/ -v -p no:cacheprovider && \
			uv run ruff check . && \
			uv run ruff format --check . && \
			uvx --from pydoclint==0.9.1 pydoclint calibre && \
			uv run pyright && \
			uv run sphinx-build -W -b html docs /tmp/calibre-docs && \
			uv build --out-dir /tmp/dist && \
			uvx twine check /tmp/dist/*'

.PHONY: granularity-run granularity granularity-check

granularity-run:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m experiments.granularity.study --jobs 4

granularity:
	uv run python -m experiments.granularity.report
	uv run jupyter execute experiments/granularity/granularity.ipynb --timeout=600 --output=build/granularity.ipynb
	latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -Werror -outdir=build experiments/granularity/note.tex

granularity-check:
	uv run pytest tests/test_granularity.py
