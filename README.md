# Agent Zero

> An autonomous agent that plays Roblox incremental games to completion.

## Project Status

🚧 **In Development** - Sprint 1

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
- [Agent Work Guide](docs/AGENT_WORK_GUIDE.md) - For parallel agent development
- [Testing Strategy](docs/TESTING_STRATEGY.md) - How to test each component

## Quick Start

```bash
# Install dependencies
make install

# Run tests
make test

# Start development environment
make dev
```

## Project Structure

```
agent-zero/
├── src/
│   ├── interfaces/     # Abstract base classes
│   ├── models/         # Pydantic data models
│   ├── vision/         # Screenshot, OCR, UI detection
│   ├── actions/        # Mouse, keyboard control
│   ├── core/           # Main agent loop
│   ├── memory/         # State persistence
│   ├── strategy/       # Decision making
│   ├── environment/    # Container management
│   └── observer/       # Web dashboard, streaming
├── tests/
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   ├── e2e/            # End-to-end tests
│   ├── performance/    # Performance benchmarks
│   └── fixtures/       # Shared test data
├── configs/            # Configuration files
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

## Development

### Prerequisites

- Python 3.11+
- Docker
- Make

### Commands

```bash
make install        # Install dependencies
make test           # Run all tests
make test-unit      # Run unit tests only
make lint           # Run linter
make typecheck      # Run mypy
make format         # Auto-format code
make coverage       # Generate coverage report
```

## Legal Notice

This project is for research and educational purposes. Using automation with Roblox may violate their Terms of Service. Use responsibly and at your own risk.

## License

MIT
