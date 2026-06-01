.PHONY: setup test lint format run up down build clean logs help db-seed

# Variables
PYTHON = python3
PIP = $(PYTHON) -m pip
COMPOSE = docker compose

help:
	@echo "Purplle Retail Intelligence - Developer Commands"
	@echo "================================================="
	@echo "setup     - Install development dependencies"
	@echo "test      - Run all pytest test suites"
	@echo "lint      - Run Ruff linter"
	@echo "format    - Run Ruff formatter"
	@echo "run       - Start the FastAPI server locally (uvicorn)"
	@echo "db-seed   - Seed the POS database natively"
	@echo "up        - Start all services via Docker Compose"
	@echo "down      - Stop all Docker Compose services"
	@echo "build     - Rebuild Docker Compose images"
	@echo "logs      - Tail Docker Compose logs"
	@echo "clean     - Remove __pycache__ and test artifacts"

setup:
	$(PIP) install -r requirements-dev.txt
	pre-commit install

test:
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v

lint:
	ruff check .

format:
	ruff check . --fix
	ruff format .

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

db-seed:
	PYTHONPATH=. $(PYTHON) -m app.ingestion

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

build:
	$(COMPOSE) build

logs:
	$(COMPOSE) logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
