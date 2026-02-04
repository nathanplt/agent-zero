# AgentZero: Roblox Game Agent - Project Plan

> **Goal**: Build an autonomous agent that plays Roblox incremental games to completion, running on its own isolated environment with user observability.

---

## Table of Contents
1. [Project Principles](#project-principles)
2. [Architecture Overview](#architecture-overview)
3. [Sprint Overview](#sprint-overview)
4. [Detailed Sprint Breakdown](#detailed-sprint-breakdown)
5. [Parallel Work Guidelines](#parallel-work-guidelines)
6. [Testing Philosophy](#testing-philosophy)
7. [Definition of Done](#definition-of-done)

---

## Project Principles

1. **Test Before Build** - Every feature has acceptance criteria and test plan defined BEFORE implementation
2. **Isolation First** - Agent runs on its own "computer", user observes remotely
3. **Incremental Verification** - Each component proven working before integration
4. **Parallel-Friendly** - Features designed for independent agent work
5. **Interface Contracts** - Clear APIs between components enable parallel development

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            USER'S MACHINE                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      OBSERVER WEB CLIENT                              │  │
│  │   [Live Screen] [Agent Logs] [Metrics] [Controls]                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WebSocket
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT ENVIRONMENT (Container/VM)                    │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  GAME RUNTIME   │  │  AGENT CORE     │  │  COMMUNICATION SERVER       │  │
│  │                 │  │                 │  │                             │  │
│  │  - Virtual      │  │  - Vision       │  │  - Screen streaming         │  │
│  │    Display      │  │  - Strategy     │  │  - Log streaming            │  │
│  │  - Roblox       │  │  - Memory       │  │  - Control API              │  │
│  │    Client       │  │  - Actions      │  │  - Metrics API              │  │
│  │  - Browser      │  │  - Orchestrator │  │                             │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         SHARED SERVICES                                 ││
│  │   [Screenshot Buffer] [State Store] [Action Queue] [Event Bus]          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Sprint Overview

| Sprint | Name | Duration | Goal | Parallelizable Features |
|--------|------|----------|------|------------------------|
| 0 | Foundation | - | Project setup, interfaces, contracts | 2 |
| 1 | Environment | - | Isolated runtime with virtual display | 3 |
| 2 | Vision | - | See and understand game state | 4 |
| 3 | Actions | - | Control game with human-like input | 3 |
| 4 | Agent Core | - | Orchestration and decision loop | 3 |
| 5 | Memory | - | Persistent state and learning | 3 |
| 6 | Strategy | - | Game-specific intelligence | 4 |
| 7 | Observer | - | User visibility and control | 4 |
| 8 | Integration | - | End-to-end operation | 2 |
| 9 | Polish | - | Optimization and reliability | 3 |

---

## Detailed Sprint Breakdown

---

### Sprint 0: Foundation

**Objective**: Establish project structure, define all interfaces, create contracts between components.

#### Feature 0.1: Project Scaffolding
**Description**: Create directory structure, dependency management, configuration system.

**Deliverables**:
- `/src` - Source code
- `/tests` - Test suites
- `/configs` - Configuration files
- `/docs` - Documentation
- `pyproject.toml` - Dependencies
- `Makefile` - Common commands

**Acceptance Criteria**:
- [ ] `make install` successfully installs all dependencies
- [ ] `make test` runs (empty) test suite
- [ ] `make lint` passes with no errors
- [ ] Configuration loads from YAML and environment variables

**Testing Strategy**:
```
TEST: Run `make install && make test && make lint`
PASS: All commands exit 0
```

**Dependencies**: None
**Parallel Work**: Yes - independent

---

#### Feature 0.2: Interface Definitions
**Description**: Define all component interfaces as abstract base classes and Pydantic models.

**Deliverables**:
- `src/interfaces/vision.py` - Vision system interface
- `src/interfaces/actions.py` - Action executor interface
- `src/interfaces/memory.py` - Memory system interface
- `src/interfaces/strategy.py` - Strategy engine interface
- `src/interfaces/environment.py` - Environment manager interface
- `src/interfaces/communication.py` - Streaming/API interface
- `src/models/` - All Pydantic data models

**Acceptance Criteria**:
- [ ] All interfaces defined with complete method signatures
- [ ] All data models defined with validation
- [ ] Type hints complete and mypy passes
- [ ] Docstrings explain purpose and usage

**Testing Strategy**:
```
TEST: mypy src/interfaces/ --strict
PASS: No type errors

TEST: Import all interfaces, instantiate mock implementations
PASS: All imports succeed, mocks instantiate
```

**Dependencies**: 0.1
**Parallel Work**: Yes - can split by interface domain

---

#### Feature 0.3: Shared Data Models
**Description**: Define all shared data structures used across components.

**Deliverables**:
```python
# src/models/game_state.py
class GameState:
    resources: dict[str, float]
    upgrades: list[Upgrade]
    current_screen: ScreenType
    ui_elements: list[UIElement]
    timestamp: datetime

# src/models/actions.py
class Action:
    type: ActionType  # CLICK, TYPE, SCROLL, WAIT
    target: Optional[Point | UIElement]
    parameters: dict

# src/models/observations.py
class Observation:
    screenshot: bytes
    game_state: GameState
    timestamp: datetime

# src/models/decisions.py
class Decision:
    reasoning: str
    action: Action
    confidence: float
    expected_outcome: str
```

**Acceptance Criteria**:
- [ ] All models serialize/deserialize to JSON
- [ ] All models have validation rules
- [ ] Models are immutable where appropriate
- [ ] 100% test coverage on model validation

**Testing Strategy**:
```
TEST: Create valid instances of each model
PASS: No validation errors

TEST: Create invalid instances (missing fields, wrong types, out-of-range)
PASS: Appropriate ValidationError raised for each case

TEST: Serialize to JSON and deserialize back
PASS: Round-trip produces equal objects
```

**Dependencies**: 0.1
**Parallel Work**: Yes - independent from 0.2

---

### Sprint 1: Environment

**Objective**: Create isolated execution environment with virtual display.

#### Feature 1.1: Container Definition
**Description**: Docker container that can run graphical applications.

**Deliverables**:
- `Dockerfile` - Container definition
- `docker-compose.yml` - Service orchestration
- Virtual display (Xvfb) configured
- VNC server for debugging
- Python environment ready

**Acceptance Criteria**:
- [ ] Container builds successfully
- [ ] Virtual display starts (Xvfb on :99)
- [ ] Can run a graphical application (e.g., `xeyes`)
- [ ] VNC connection shows the virtual display
- [ ] Python 3.11+ available with pip

**Testing Strategy**:
```
TEST: docker build -t agentzero .
PASS: Build completes without errors

TEST: docker run agentzero xeyes &
      Connect via VNC to container:5900
PASS: xeyes window visible in VNC

TEST: docker run agentzero python --version
PASS: Python 3.11+ version printed
```

**Dependencies**: 0.1
**Parallel Work**: Yes - independent

---

#### Feature 1.2: Browser Runtime
**Description**: Chromium browser running in container for Roblox web player.

**Deliverables**:
- Chromium installed in container
- Browser automation ready (Playwright)
- Can navigate to Roblox

**Acceptance Criteria**:
- [ ] Chromium launches in virtual display
- [ ] Playwright can control browser
- [ ] Can navigate to roblox.com
- [ ] Page renders correctly (screenshot verification)

**Testing Strategy**:
```
TEST: Launch Chromium via Playwright, navigate to example.com
PASS: Screenshot contains expected content

TEST: Navigate to roblox.com
PASS: Roblox homepage loads (verify logo in screenshot)

TEST: Take screenshot, verify dimensions match virtual display
PASS: Screenshot is 1920x1080 (or configured size)
```

**Dependencies**: 1.1
**Parallel Work**: No - requires 1.1 complete

---

#### Feature 1.3: Environment Manager
**Description**: Python API to start/stop/manage the container environment.

**Deliverables**:
- `src/environment/manager.py` - Environment lifecycle
- `src/environment/display.py` - Virtual display control
- `src/environment/browser.py` - Browser lifecycle

**Acceptance Criteria**:
- [ ] `EnvironmentManager.start()` launches container
- [ ] `EnvironmentManager.stop()` cleanly shuts down
- [ ] `EnvironmentManager.status()` returns health info
- [ ] `EnvironmentManager.screenshot()` returns current frame
- [ ] Handles crashes gracefully with auto-restart

**Testing Strategy**:
```
TEST: start() -> status() -> stop()
PASS: Status shows "running", stop succeeds

TEST: start() -> screenshot()
PASS: Returns valid image bytes, correct dimensions

TEST: start() -> kill container externally -> status()
PASS: Status shows "crashed" or auto-restart triggers

TEST: start() -> start() (double start)
PASS: Second start is no-op or raises appropriate error
```

**Dependencies**: 1.1, 1.2, 0.2
**Parallel Work**: Yes - interface defined in 0.2

---

### Sprint 2: Vision

**Objective**: Extract structured game state from screenshots.

#### Feature 2.1: Screenshot Capture
**Description**: Fast, reliable screenshot capture from virtual display.

**Deliverables**:
- `src/vision/capture.py` - Screenshot capture
- Configurable frame rate
- Screenshot buffer for temporal analysis

**Acceptance Criteria**:
- [ ] Captures at 10+ FPS
- [ ] Returns raw bytes and PIL Image
- [ ] Buffer stores last N frames
- [ ] Timestamps accurate to millisecond

**Testing Strategy**:
```
TEST: Capture 100 frames, measure time
PASS: Average < 100ms per frame (10+ FPS)

TEST: Capture with visual change between frames
PASS: Frames are different (pixel comparison)

TEST: Check buffer after 20 captures with buffer size 10
PASS: Buffer contains exactly 10 most recent frames
```

**Dependencies**: 1.3
**Parallel Work**: Yes - uses Environment interface

---

#### Feature 2.2: OCR System
**Description**: Extract text and numbers from screenshots.

**Deliverables**:
- `src/vision/ocr.py` - Text extraction
- Region-based OCR (extract from specific areas)
- Number parsing with unit handling (1.5K, 2.3M, etc.)

**Acceptance Criteria**:
- [ ] Extracts visible text with 95%+ accuracy
- [ ] Handles game fonts (may need training)
- [ ] Parses abbreviated numbers correctly
- [ ] Returns text with bounding boxes

**Testing Strategy**:
```
TEST: OCR on test image with known text
PASS: Extracted text matches expected (Levenshtein distance < 5%)

TEST: Parse "1.5K" -> 1500, "2.3M" -> 2300000
PASS: All number formats parsed correctly

TEST: Region-based OCR on resource counter area
PASS: Returns only text from specified region
```

**Test Fixtures**:
- `tests/fixtures/ocr/` - Screenshots with known text
- `tests/fixtures/ocr/numbers.json` - Number format test cases

**Dependencies**: 2.1
**Parallel Work**: Yes - independent module

---

#### Feature 2.3: UI Element Detection
**Description**: Detect buttons, menus, clickable elements.

**Deliverables**:
- `src/vision/ui_detection.py` - Element detection
- Confidence scores for detections
- Element type classification

**Acceptance Criteria**:
- [ ] Detects buttons with 90%+ recall
- [ ] Returns bounding boxes for each element
- [ ] Classifies element types (button, menu, resource, etc.)
- [ ] Handles overlapping UI gracefully

**Testing Strategy**:
```
TEST: Detection on annotated test screenshots
PASS: IoU > 0.7 for 90% of labeled elements

TEST: Detection returns confidence scores
PASS: All detections have confidence in [0, 1]

TEST: Classification accuracy on test set
PASS: 85%+ elements correctly classified
```

**Test Fixtures**:
- `tests/fixtures/ui/` - Annotated screenshots with element labels

**Dependencies**: 2.1
**Parallel Work**: Yes - independent from 2.2

---

#### Feature 2.4: LLM Vision Integration
**Description**: Use vision LLM (Claude/GPT-4V) for complex understanding.

**Deliverables**:
- `src/vision/llm_vision.py` - LLM integration
- Prompt templates for game state extraction
- Structured output parsing
- Set-of-Mark annotation

**Acceptance Criteria**:
- [ ] Sends screenshot to vision LLM
- [ ] Returns structured GameState object
- [ ] Handles API errors gracefully
- [ ] Caches repeated identical frames
- [ ] Set-of-Mark improves grounding accuracy

**Testing Strategy**:
```
TEST: Send test screenshot, parse response to GameState
PASS: Valid GameState returned with expected fields

TEST: Send same screenshot twice
PASS: Second call uses cache (no API call)

TEST: Mock API error, verify retry logic
PASS: Retries 3 times with backoff, then raises

TEST: Compare accuracy with/without Set-of-Mark
PASS: Set-of-Mark improves element reference accuracy by 20%+
```

**Dependencies**: 2.1, 0.3
**Parallel Work**: Yes - independent from 2.2, 2.3

---

### Sprint 3: Actions

**Objective**: Control the game with human-like input.

#### Feature 3.1: Mouse Control
**Description**: Move and click mouse in virtual display.

**Deliverables**:
- `src/actions/mouse.py` - Mouse control
- Human-like movement (Bezier curves)
- Click variations (single, double, hold)
- Movement timing variance

**Acceptance Criteria**:
- [ ] Move to any coordinate in display
- [ ] Click accurately on target
- [ ] Movement follows curved path (not linear)
- [ ] Timing varies naturally (not robotic)

**Testing Strategy**:
```
TEST: Move to 10 random points, capture mouse position
PASS: Final positions within 2px of targets

TEST: Record movement path, verify non-linear
PASS: Path has curvature (not straight line)

TEST: Measure 100 click timings
PASS: Standard deviation > 20ms (has variance)

TEST: Click button in test app, verify activation
PASS: Button state changes to "pressed"
```

**Dependencies**: 1.3
**Parallel Work**: Yes - uses Environment interface

---

#### Feature 3.2: Keyboard Control
**Description**: Type text and send key commands.

**Deliverables**:
- `src/actions/keyboard.py` - Keyboard control
- Text typing with natural speed
- Special keys (Enter, Escape, Tab, etc.)
- Key combinations (Ctrl+A, etc.)

**Acceptance Criteria**:
- [ ] Type any ASCII text
- [ ] Special keys work correctly
- [ ] Key combinations work
- [ ] Typing speed varies naturally

**Testing Strategy**:
```
TEST: Type "Hello World" into text input, read back
PASS: Input contains "Hello World"

TEST: Press Escape, verify effect in test app
PASS: App responds to Escape (e.g., closes dialog)

TEST: Press Ctrl+A, verify select all
PASS: All text selected in input field

TEST: Measure character timing over 100 chars
PASS: Timing varies (50-150ms range, not constant)
```

**Dependencies**: 1.3
**Parallel Work**: Yes - parallel with 3.1

---

#### Feature 3.3: Action Executor
**Description**: Unified action execution with validation.

**Deliverables**:
- `src/actions/executor.py` - Action execution
- Pre-action validation
- Post-action verification
- Action queuing and rate limiting

**Acceptance Criteria**:
- [ ] Executes Action objects from models
- [ ] Validates target exists before click
- [ ] Verifies action had effect (state changed)
- [ ] Rate limits to avoid detection

**Testing Strategy**:
```
TEST: Execute valid click action
PASS: Click performed, state changed

TEST: Execute click on non-existent element
PASS: ActionError raised before attempt

TEST: Execute action, verify post-condition
PASS: Verification confirms expected change

TEST: Rapid-fire 100 actions
PASS: Rate limiting enforces minimum delay
```

**Dependencies**: 3.1, 3.2, 0.3, 2.3
**Parallel Work**: No - integrates 3.1 and 3.2

---

### Sprint 4: Agent Core

**Objective**: Core orchestration and decision loop.

#### Feature 4.1: Observation Pipeline
**Description**: Capture → Process → Structured State.

**Deliverables**:
- `src/core/observation.py` - Observation pipeline
- Combines vision components
- Produces Observation objects

**Acceptance Criteria**:
- [ ] Single call produces complete Observation
- [ ] Includes screenshot, game state, timestamp
- [ ] Handles vision errors gracefully
- [ ] Performance < 2 seconds per observation

**Testing Strategy**:
```
TEST: Call observe() with game running
PASS: Returns valid Observation with all fields

TEST: Measure observation latency over 50 calls
PASS: 95th percentile < 2 seconds

TEST: Inject vision error, verify handling
PASS: Returns partial observation or raises clean error
```

**Dependencies**: 2.1, 2.2, 2.3, 2.4
**Parallel Work**: No - integrates vision components

---

#### Feature 4.2: Decision Engine
**Description**: ReAct-style reasoning and action selection.

**Deliverables**:
- `src/core/decision.py` - Decision making
- ReAct prompting
- Chain-of-thought reasoning
- Action selection with confidence

**Acceptance Criteria**:
- [ ] Takes Observation, returns Decision
- [ ] Decision includes reasoning trace
- [ ] Confidence scores are calibrated
- [ ] Handles ambiguous states gracefully

**Testing Strategy**:
```
TEST: Decision on clear game state (obvious next action)
PASS: Correct action selected, high confidence

TEST: Decision on ambiguous state
PASS: Reasonable action, lower confidence

TEST: Verify reasoning trace is coherent
PASS: Reasoning explains the decision logically

TEST: Mock 100 decisions, check confidence calibration
PASS: High-confidence decisions correct more often
```

**Dependencies**: 0.3, 4.1
**Parallel Work**: Yes - uses Observation interface

---

#### Feature 4.3: Main Loop
**Description**: Observe → Decide → Act → Repeat.

**Deliverables**:
- `src/core/loop.py` - Main agent loop
- Configurable loop rate
- Error recovery
- Graceful shutdown

**Acceptance Criteria**:
- [ ] Runs continuously until stopped
- [ ] Each iteration: observe, decide, act
- [ ] Recovers from transient errors
- [ ] Logs each iteration for debugging
- [ ] Clean shutdown on signal

**Testing Strategy**:
```
TEST: Run loop for 10 iterations, verify progression
PASS: All 10 iterations complete, game state progresses

TEST: Inject error on iteration 5
PASS: Error logged, loop continues

TEST: Send SIGTERM during loop
PASS: Clean shutdown, no orphan processes

TEST: Verify logs contain observation, decision, action
PASS: All components logged each iteration
```

**Dependencies**: 4.1, 4.2, 3.3
**Parallel Work**: No - integrates core components

---

### Sprint 5: Memory

**Objective**: Persistent state and learning from experience.

#### Feature 5.1: Game State Persistence
**Description**: Save and restore game state across sessions.

**Deliverables**:
- `src/memory/persistence.py` - State persistence
- SQLite backend
- State snapshots with timestamps

**Acceptance Criteria**:
- [ ] Save current game state
- [ ] Restore state on restart
- [ ] Query historical states
- [ ] Handle corrupted data gracefully

**Testing Strategy**:
```
TEST: Save state, stop, restore, compare
PASS: Restored state matches saved

TEST: Save 1000 states, query last 10
PASS: Returns correct 10 most recent

TEST: Corrupt database, attempt load
PASS: Graceful error, can reinitialize
```

**Dependencies**: 0.3
**Parallel Work**: Yes - independent module

---

#### Feature 5.2: Episodic Memory
**Description**: Remember what happened and what worked.

**Deliverables**:
- `src/memory/episodic.py` - Episode tracking
- Action-outcome pairs
- Success/failure annotations
- Similarity search for relevant episodes

**Acceptance Criteria**:
- [ ] Record action → outcome pairs
- [ ] Annotate success/failure
- [ ] Query similar past situations
- [ ] Prune old/irrelevant memories

**Testing Strategy**:
```
TEST: Record 100 episodes, query similar to current
PASS: Returns relevant episodes (manual verification)

TEST: Record success and failure episodes
PASS: Success/failure correctly distinguished

TEST: Memory reaches limit, verify pruning
PASS: Oldest/least relevant pruned, size stable
```

**Dependencies**: 5.1, 0.3
**Parallel Work**: Yes - parallel with 5.3

---

#### Feature 5.3: Strategy Memory
**Description**: Remember high-level strategies and patterns.

**Deliverables**:
- `src/memory/strategy.py` - Strategy tracking
- Strategy effectiveness scores
- Pattern recognition

**Acceptance Criteria**:
- [ ] Store named strategies with descriptions
- [ ] Track success rate per strategy
- [ ] Identify patterns from episodes
- [ ] Recommend strategies for situations

**Testing Strategy**:
```
TEST: Record strategy use and outcomes
PASS: Effectiveness scores calculated correctly

TEST: Query recommended strategy for game state
PASS: Returns highest-rated applicable strategy

TEST: Inject clear pattern, verify detection
PASS: Pattern identified and named
```

**Dependencies**: 5.2
**Parallel Work**: Yes - parallel with 5.2 after 5.1

---

### Sprint 6: Strategy

**Objective**: Game-specific intelligence and optimization.

#### Feature 6.1: Goal Hierarchy
**Description**: Long-term goals broken into subgoals.

**Deliverables**:
- `src/strategy/goals.py` - Goal management
- Goal tree structure
- Progress tracking
- Dynamic replanning

**Acceptance Criteria**:
- [ ] Define hierarchical goals
- [ ] Track progress on each goal
- [ ] Detect goal completion
- [ ] Replan when blocked

**Testing Strategy**:
```
TEST: Define goal tree, simulate progress
PASS: Subgoals complete, parent goal completes

TEST: Block a subgoal, verify replanning
PASS: Alternative path found or goal marked blocked

TEST: Verify progress tracking accuracy
PASS: Progress percentages match actual state
```

**Dependencies**: 0.3, 5.1
**Parallel Work**: Yes - independent module

---

#### Feature 6.2: Incremental Game Meta-Strategy
**Description**: Knowledge about optimal incremental game progression.

**Deliverables**:
- `src/strategy/incremental.py` - Incremental game knowledge
- Prestige timing optimization
- Resource allocation strategies
- Upgrade prioritization

**Acceptance Criteria**:
- [ ] Recommends optimal prestige timing
- [ ] Allocates resources efficiently
- [ ] Prioritizes upgrades correctly
- [ ] Adapts to game-specific mechanics

**Testing Strategy**:
```
TEST: Given resource state, recommend allocation
PASS: Allocation matches optimal strategy

TEST: Simulate prestige scenarios
PASS: Recommended timing maximizes long-term progress

TEST: Upgrade prioritization on test game state
PASS: Highest ROI upgrades selected first
```

**Dependencies**: 6.1, 0.3
**Parallel Work**: Yes - parallel with 6.3

---

#### Feature 6.3: Planning System
**Description**: Multi-step lookahead and planning.

**Deliverables**:
- `src/strategy/planning.py` - Planning system
- State prediction
- Plan evaluation
- Backtracking on failure

**Acceptance Criteria**:
- [ ] Generate multi-step plans
- [ ] Evaluate plan expected value
- [ ] Execute plans with checkpoints
- [ ] Backtrack when plan fails

**Testing Strategy**:
```
TEST: Generate plan for simple scenario
PASS: Plan achieves goal in simulation

TEST: Execute plan, inject failure at step 3
PASS: Backtrack occurs, replanning succeeds

TEST: Compare plan evaluation to actual outcome
PASS: Correlation > 0.7 between predicted and actual value
```

**Dependencies**: 6.1, 4.2
**Parallel Work**: Yes - parallel with 6.2

---

#### Feature 6.4: Game Adapter
**Description**: Specific adapter for target Roblox game.

**Deliverables**:
- `src/strategy/adapters/target_game.py` - Game adapter
- Game-specific UI mappings
- Game-specific strategies
- Win condition detection

**Acceptance Criteria**:
- [ ] Maps game UI to standard elements
- [ ] Implements game-specific actions
- [ ] Detects game completion
- [ ] Handles game-specific edge cases

**Testing Strategy**:
```
TEST: UI mapping on game screenshots
PASS: 95% of elements correctly mapped

TEST: Game-specific action execution
PASS: Actions work correctly in game

TEST: Win condition detection
PASS: Correctly identifies game completion
```

**Note**: Requires target game selection before implementation.

**Dependencies**: 6.1, 6.2, 6.3, 2.3
**Parallel Work**: No - requires other strategy components

---

### Sprint 7: Observer

**Objective**: User visibility and control interface.

#### Feature 7.1: Screen Streaming
**Description**: Stream agent's screen to user.

**Deliverables**:
- `src/observer/streaming.py` - Screen streaming
- WebSocket-based frame streaming
- Configurable quality/framerate
- Low latency (<500ms)

**Acceptance Criteria**:
- [ ] Streams screen at 10+ FPS
- [ ] Latency < 500ms
- [ ] Quality adjustable
- [ ] Handles reconnection gracefully

**Testing Strategy**:
```
TEST: Connect client, verify frames received
PASS: Frames arrive at 10+ FPS

TEST: Measure end-to-end latency
PASS: 95th percentile < 500ms

TEST: Disconnect and reconnect
PASS: Streaming resumes within 2 seconds
```

**Dependencies**: 2.1
**Parallel Work**: Yes - uses Vision interface

---

#### Feature 7.2: Log Streaming
**Description**: Stream agent reasoning to user.

**Deliverables**:
- `src/observer/logs.py` - Log streaming
- Real-time decision logs
- Structured log format
- Log levels (debug, info, decision)

**Acceptance Criteria**:
- [ ] Logs stream in real-time
- [ ] Decision reasoning visible
- [ ] Log levels filterable
- [ ] No sensitive data leaked

**Testing Strategy**:
```
TEST: Trigger agent decision, verify log appears
PASS: Decision log received within 1 second

TEST: Filter by log level
PASS: Only matching levels shown

TEST: Verify no API keys in logs
PASS: Sensitive data redacted
```

**Dependencies**: 4.3
**Parallel Work**: Yes - parallel with 7.1

---

#### Feature 7.3: Control API
**Description**: User can control agent (start, stop, configure).

**Deliverables**:
- `src/observer/control.py` - Control API
- REST API for commands
- Start/stop/pause controls
- Configuration updates

**Acceptance Criteria**:
- [ ] Start agent via API
- [ ] Stop agent cleanly via API
- [ ] Pause/resume agent
- [ ] Update configuration at runtime

**Testing Strategy**:
```
TEST: POST /start, verify agent starts
PASS: Agent running, status endpoint confirms

TEST: POST /stop, verify clean shutdown
PASS: Agent stopped, no orphan processes

TEST: POST /config with new setting
PASS: Agent behavior reflects new config
```

**Dependencies**: 4.3
**Parallel Work**: Yes - parallel with 7.1, 7.2

---

#### Feature 7.4: Web Dashboard
**Description**: Web UI for observation and control.

**Deliverables**:
- `src/observer/web/` - Web dashboard
- Live screen viewer
- Log viewer
- Metrics display
- Control buttons

**Acceptance Criteria**:
- [ ] Single-page web app
- [ ] Shows live screen
- [ ] Shows scrolling logs
- [ ] Control buttons work
- [ ] Works in modern browsers

**Testing Strategy**:
```
TEST: Load dashboard, verify components render
PASS: Screen, logs, controls all visible

TEST: Click start button, verify effect
PASS: Agent starts, UI updates

TEST: Run for 5 minutes, verify stability
PASS: No memory leaks, no crashes
```

**Dependencies**: 7.1, 7.2, 7.3
**Parallel Work**: No - integrates observer components

---

### Sprint 8: Integration

**Objective**: End-to-end system working together.

#### Feature 8.1: System Integration
**Description**: All components working together.

**Deliverables**:
- Integration tests
- Docker Compose for full stack
- Startup/shutdown orchestration

**Acceptance Criteria**:
- [ ] `docker-compose up` starts everything
- [ ] Agent plays game visible in dashboard
- [ ] All features work together
- [ ] Clean shutdown of all components

**Testing Strategy**:
```
TEST: docker-compose up, verify all services healthy
PASS: All containers running, health checks pass

TEST: Run agent for 1 hour, monitor stability
PASS: No crashes, continuous operation

TEST: docker-compose down, verify clean shutdown
PASS: All containers stopped, no orphans
```

**Dependencies**: All previous sprints
**Parallel Work**: No - integration phase

---

#### Feature 8.2: End-to-End Testing
**Description**: Automated tests of full agent operation.

**Deliverables**:
- `tests/e2e/` - End-to-end tests
- Game progression tests
- Error recovery tests
- Performance benchmarks

**Acceptance Criteria**:
- [ ] Agent can start game from scratch
- [ ] Agent progresses through early game
- [ ] Agent recovers from common errors
- [ ] Performance meets targets

**Testing Strategy**:
```
TEST: Start agent on new game, run 30 minutes
PASS: Measurable progress made

TEST: Inject disconnect, verify recovery
PASS: Agent reconnects and continues

TEST: Measure actions per minute
PASS: APM > 10 (configurable threshold)
```

**Dependencies**: 8.1
**Parallel Work**: No - requires full integration

---

### Sprint 9: Polish

**Objective**: Optimization, reliability, documentation.

#### Feature 9.1: Performance Optimization
**Description**: Optimize for speed and efficiency.

**Deliverables**:
- Profiling reports
- Optimized hot paths
- Reduced API calls
- Lower latency

**Acceptance Criteria**:
- [ ] 50% reduction in API costs
- [ ] Loop latency < 1 second
- [ ] Memory stable over 24 hours
- [ ] CPU usage < 50% baseline

**Testing Strategy**:
```
TEST: Profile 1000 loop iterations
PASS: Identify and optimize top 3 bottlenecks

TEST: Run 24 hours, monitor memory
PASS: Memory growth < 100MB

TEST: Compare API costs before/after
PASS: 50%+ reduction achieved
```

**Dependencies**: 8.1
**Parallel Work**: Yes - independent optimization

---

#### Feature 9.2: Error Handling & Recovery
**Description**: Robust handling of all error cases.

**Deliverables**:
- Error taxonomy
- Recovery strategies per error type
- Alerting for unrecoverable errors

**Acceptance Criteria**:
- [ ] All error types classified
- [ ] Automatic recovery where possible
- [ ] User notified of critical errors
- [ ] No silent failures

**Testing Strategy**:
```
TEST: Inject each error type, verify recovery
PASS: Recoverable errors recovered, others alerted

TEST: Chaos testing (random failures)
PASS: System remains stable, recovers appropriately
```

**Dependencies**: 8.1
**Parallel Work**: Yes - parallel with 9.1

---

#### Feature 9.3: Documentation
**Description**: Complete project documentation.

**Deliverables**:
- `docs/architecture.md` - System architecture
- `docs/setup.md` - Setup guide
- `docs/development.md` - Development guide
- `docs/api.md` - API reference
- Inline code documentation

**Acceptance Criteria**:
- [ ] New developer can set up in < 30 minutes
- [ ] All public APIs documented
- [ ] Architecture clearly explained
- [ ] Troubleshooting guide included

**Testing Strategy**:
```
TEST: Fresh developer follows setup guide
PASS: Successfully running within 30 minutes

TEST: All public functions have docstrings
PASS: Documentation coverage > 90%
```

**Dependencies**: 8.1
**Parallel Work**: Yes - parallel with 9.1, 9.2

---

## Parallel Work Guidelines

### Component Independence Map

```
Sprint 0: [0.1] ──┬──> [0.2]
                 └──> [0.3]

Sprint 1: [1.1] ──> [1.2] ──> [1.3]

Sprint 2: [2.1] ──┬──> [2.2]
                 ├──> [2.3]
                 └──> [2.4]

Sprint 3: [3.1] ──┐
                 ├──> [3.3]
          [3.2] ──┘

Sprint 4: [4.1] ──> [4.2] ──> [4.3]

Sprint 5: [5.1] ──┬──> [5.2]
                 └──> [5.3]

Sprint 6: [6.1] ──┬──> [6.2] ──┐
                 └──> [6.3] ──┼──> [6.4]
                              │
Sprint 7: [7.1] ──┐           │
          [7.2] ──┼──> [7.4]  │
          [7.3] ──┘           │
                              │
Sprint 8: [8.1] <─────────────┘
             │
             v
          [8.2]

Sprint 9: [9.1]
          [9.2]
          [9.3]
```

### Rules for Parallel Agent Work

1. **Interface First**: Agents MUST implement against interfaces defined in Sprint 0
2. **Test Fixtures Shared**: All agents use shared test fixtures in `tests/fixtures/`
3. **No Cross-Component Imports**: Components import interfaces, not implementations
4. **PR per Feature**: Each feature is a separate branch and PR
5. **Integration Points**: Features marked "No - integrates" must wait for dependencies

### Assigning Work to Agents

| Agent | Can Work On |
|-------|-------------|
| Agent A | 0.1, 1.1, 2.2, 3.1, 5.1, 7.1 |
| Agent B | 0.2, 2.3, 3.2, 5.2, 7.2 |
| Agent C | 0.3, 2.4, 6.1, 6.2, 7.3 |
| Agent D | 1.2, 1.3, 4.x (sequential), 6.3 |

---

## Testing Philosophy

### Test Pyramid

```
         /\
        /  \      E2E Tests (Sprint 8)
       /    \     - Full agent operation
      /──────\    
     /        \   Integration Tests
    /          \  - Component combinations
   /────────────\ 
  /              \ Unit Tests
 /                \ - Individual functions
/──────────────────\ - Each feature has unit tests
```

### Test Requirements

1. **Unit Tests**: Every feature has unit tests BEFORE implementation
2. **Test Fixtures**: Shared fixtures for consistent testing
3. **Mocking**: External services (LLM APIs) are mocked in tests
4. **Coverage**: Minimum 80% code coverage
5. **Performance Tests**: Latency and throughput benchmarks

### Test-First Workflow

```
1. Read feature specification
2. Write test file with all test cases (they will fail)
3. Implement feature until tests pass
4. Refactor if needed (tests still pass)
5. PR includes tests + implementation
```

---

## Definition of Done

A feature is DONE when:

- [ ] All acceptance criteria met
- [ ] All specified tests pass
- [ ] Code coverage > 80%
- [ ] No linter errors
- [ ] Type hints complete (mypy passes)
- [ ] Docstrings on public functions
- [ ] PR reviewed and approved
- [ ] Merged to main branch

---

## Appendix: Target Game Selection

Before Sprint 6.4, we must select a target Roblox incremental game. Criteria:

1. **Publicly available** (not private/invite-only)
2. **Pure incremental** (click/upgrade loop, minimal action requirements)
3. **Reasonable scope** (completable in hours, not months)
4. **Stable** (not frequently updated, breaking automation)

Candidates to evaluate:
- Clicker Simulator
- Tapping Legends
- Clicking Legends
- Mining Simulator (incremental aspects)

Selection should happen before Sprint 6 begins.

---

## Next Steps

1. **Approve this plan** or request modifications
2. **Initialize Sprint 0** - Create project scaffolding
3. **Assign parallel features** to multiple agents
4. **Begin Sprint 0.1, 0.2, 0.3 in parallel**
