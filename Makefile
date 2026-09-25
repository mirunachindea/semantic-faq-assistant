.PHONY: install lint format typecheck test check run up down logs sync evaluate

install:        ## Install runtime + dev dependencies and git hooks
	uv sync
	uv run pre-commit install

format:         ## Auto-format and fix lint
	uv run ruff format .
	uv run ruff check --fix .

lint:           ## Lint (no changes)
	uv run ruff format --check .
	uv run ruff check .

typecheck:      ## Static type checking (strict)
	uv run mypy src tests

test:           ## Run the test suite (no network, no API key needed)
	uv run pytest

check: lint typecheck test  ## Everything CI runs

run:            ## Run the API locally with auto-reload
	uv run uvicorn faq_assistant.main:create_app --factory --reload

up:             ## Start postgres, redis, migrate job, API and worker
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api worker

sync:           ## Incrementally embed the knowledge base
	uv run faq-admin sync

evaluate:       ## Retrieval metrics + threshold sweep (2 embedding requests)
	uv run faq-admin evaluate
