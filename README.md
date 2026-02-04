# Agent Zero

> An autonomous agent that plays Roblox incremental games to completion.

## Project Status

**In Development** - Sprints 0-2 complete, Sprint 3 in progress

## Overview

Agent Zero is a general-purpose computer-using agent, starting with mastery of Roblox incremental games. The agent runs in its own isolated environment (container/VM), and users observe its progress through a web dashboard.

### Key Features (Planned)

- **Vision System**: Understands game state from screenshots using OCR and LLM vision
- **Human-like Actions**: Controls mouse and keyboard with natural timing and movement
- **Strategic Planning**: Makes intelligent decisions about game progression
- **Memory & Learning**: Remembers what works and improves over time
- **User Observability**: Watch the agent play in real-time through web dashboard

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER'S MACHINE                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   OBSERVER WEB CLIENT                     │  │
│  │   [Live Screen] [Agent Logs] [Metrics] [Controls]         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 AGENT ENVIRONMENT (Container)                   │
│  [Game Runtime] ←→ [Agent Brain] ←→ [Communication Server]     │
└─────────────────────────────────────────────────────────────────┘
```

## Documentation

- [Project Plan](PROJECT_PLAN.md) - Detailed sprint and feature breakdown
- [Roadmap](ROADMAP.md) - Visual progress tracker
- [Status](STATUS.md) - Current work and completed features
- [Agent Work Guide](docs/AGENT_WORK_GUIDE.md) - For parallel agent development
- [Testing Strategy](docs/TESTING_STRATEGY.md) - How to test each component

## Quick Start

```bash
# Install dependencies
make install

# Install development dependencies
make install-dev

# Run tests
make test

# Run all checks (lint + typecheck + test)
make check
```

## Project Structure

```
agent-zero/
├── src/
│   ├── interfaces/     # Abstract base classes and contracts
│   ├── models/         # Pydantic data models
│   ├── vision/         # Screenshot, OCR, UI detection, LLM vision
│   ├── actions/        # Mouse, keyboard control, input backends
│   ├── environment/    # Display, browser, auth, environment manager
│   └── config/         # YAML + env var configuration loader
├── tests/              # Test suite (unit + integration)
│   └── fixtures/       # Shared test data (OCR numbers, UI elements)
├── configs/            # Configuration files (default.yaml)
├── docker/             # Docker entrypoint and display scripts
├── docs/               # Documentation guides
├── Dockerfile          # Container definition
├── docker-compose.yml  # Service orchestration
├── Makefile            # Development commands
└── pyproject.toml      # Project metadata and dependencies
```

## Development

### Prerequisites

- Python 3.11+
- Docker (for container builds)
- Make

### Commands

```bash
make install        # Install production dependencies
make install-dev    # Install development dependencies
make test           # Run all tests
make test-cov       # Run tests with coverage report
make lint           # Run linter (ruff)
make typecheck      # Run type checker (mypy)
make format         # Auto-format code (ruff)
make check          # Run all checks (lint + typecheck + test)
make docker-build   # Build Docker image
make docker-run     # Run container with VNC
```

## Legal Notice

This project is for research and educational purposes. Using automation with Roblox may violate their Terms of Service. Use responsibly and at your own risk.

## License

MIT
