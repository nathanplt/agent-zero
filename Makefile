.PHONY: install install-dev test lint typecheck format clean help docker-build docker-run docker-test

# Default Python interpreter
PYTHON ?= python3

# Docker image name
DOCKER_IMAGE ?= agentzero

# Help target
help:
	@echo "AgentZero Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test         Run test suite"
	@echo "  make test-cov     Run tests with coverage report"
	@echo "  make lint         Run linter (ruff)"
	@echo "  make typecheck    Run type checker (mypy)"
	@echo "  make format       Format code (ruff)"
	@echo "  make check        Run all checks (lint + typecheck + test)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build Build the Docker image"
	@echo "  make docker-run   Run the container with VNC"
	@echo "  make docker-test  Run Docker-specific tests"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Remove build artifacts and caches"

# Installation targets
install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

install-all:
	$(PYTHON) -m pip install -e ".[all]"

# Test targets
test:
	$(PYTHON) -m pytest tests/

test-cov:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# Linting and formatting
lint:
	$(PYTHON) -m ruff check src/ tests/

lint-fix:
	$(PYTHON) -m ruff check src/ tests/ --fix

format:
	$(PYTHON) -m ruff format src/ tests/

# Type checking
typecheck:
	$(PYTHON) -m mypy src/

# Run all checks
check: lint typecheck test

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Docker targets
docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-run:
	docker run -it --rm -p 5900:5900 $(DOCKER_IMAGE)

docker-test:
	$(PYTHON) -m pytest tests/test_container.py -v

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down
