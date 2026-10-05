# Developer entry points. Run from the repo root inside WSL2.
#   make install    create .venv and install pinned deps (torch from the CUDA index)
#   make env        print hardware / library versions and run GPU smoke tests
#   make test       full test suite (GPU + model downloads)
#   make test-fast  only fast tests: no GPU required, no model downloads
#   make lint       static checks

PYTHON  ?= python3
VENV    := .venv
BIN     := $(VENV)/bin
TORCH_INDEX := https://download.pytorch.org/whl/cu128

.PHONY: install env test test-fast lint format lock clean

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: $(BIN)/python
	$(BIN)/pip install torch==2.11.0 --index-url $(TORCH_INDEX)
	$(BIN)/pip install -e ".[dev]"

env:
	$(BIN)/python scripts/env_check.py

test:
	$(BIN)/pytest -v

test-fast:
	$(BIN)/pytest -v -m "not slow and not gpu"

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

# Full transitive snapshot of the working environment (pyproject pins only top-level deps).
lock:
	$(BIN)/pip freeze --exclude-editable > requirements.lock

clean:
	rm -rf .pytest_cache .ruff_cache .hypothesis build *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
