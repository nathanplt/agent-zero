# Agent Zero: Roblox Game Agent - Project Plan

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
│  │                   OBSERVER WEB CLIENT (React SPA)                     │  │
│  │   [Live Screen] [Agent Logs] [Metrics] [Controls]                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WebSocket + REST (FastAPI)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT ENVIRONMENT (Container/VM)                    │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  GAME RUNTIME   │  │  AGENT CORE     │  │  COMMUNICATION SERVER       │  │
│  │                 │  │                 │  │  (FastAPI + WebSockets)     │  │
│  │  - Virtual      │  │  - Vision       │  │                             │  │
│  │    Display      │  │  - Strategy     │  │  - Screen streaming         │  │
│  │  - Roblox       │  │  - Memory       │  │  - Log streaming            │  │
│  │    Client       │  │  - Actions      │  │  - Control API              │  │
│  │  - Browser      │  │  - Orchestrator │  │  - Metrics API              │  │
│  │  - Auth/Session │  │  - Metrics      │  │                             │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         SHARED SERVICES                                 ││
│  │   [Screenshot Buffer (2.1)] [State Store (5.1)] [Metrics Store (4.3)]   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         INPUT LAYER                                     ││
│  │   [InputBackend Interface] ──> [PlaywrightInputBackend]                 ││
│  │   MouseController ─────────┘                                            ││
│  │   KeyboardController ──────┘                                            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Architecture Decisions

1. **Input Layer**: Controllers contain human-like timing logic; `InputBackend` interface allows swappable backends (Playwright for browser, pyautogui for desktop)
2. **Web Stack**: FastAPI for REST/WebSocket backend, React SPA for dashboard
3. **LLM Usage**: Single LLM client shared between Vision (2.4) and Decision Engine (4.2)
4. **Metrics**: Collected in main loop (4.3), stored in memory, streamed via Observer (7.x)

---

## Sprint Overview

| Sprint | Name | Duration | Goal | Features |
|--------|------|----------|------|----------|
| 0 | Foundation | - | Project setup, interfaces, contracts | 0.1, 0.2, 0.3 |
| 1 | Environment | - | Isolated runtime with virtual display and auth | 1.1, 1.2, 1.3, **1.4** |
| 2 | Vision | - | See and understand game state | 2.1, 2.2, 2.3, 2.4 |
| 3 | Actions | - | Control game with human-like input | 3.1, 3.2, **3.3** |
| 4 | Agent Core | - | Orchestration and decision loop | 4.1, 4.2, 4.3 |
| 5 | Memory | - | Persistent state and learning | 5.1, 5.2, 5.3 |
| 6 | Strategy | - | Game-specific intelligence | **6.0**, 6.1, 6.2, 6.3, 6.4 |
| 7 | Observer | - | User visibility and control | 7.1, 7.2, 7.3, 7.4 |
| 8 | Integration | - | End-to-end operation | 8.1, 8.2 |
| 9 | Polish | - | Optimization and reliability | 9.1, 9.2, 9.3 |

**New/Modified Features (bold):**
- **1.4**: Roblox Authentication - Login, session management, 2FA handling
- **3.3**: Action Executor (expanded) - Now includes InputBackend integration with Playwright
- **6.0**: Target Game Selection - Evaluate and document target game before building adapters

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
- `/frontend` - React dashboard (created in 7.4)
- `pyproject.toml` - Dependencies
- `Makefile` - Common commands

**Configuration System**:
```python
# src/config/loader.py
# Configuration is loaded once at startup and passed to components
# Runtime updates are handled via the ConfigManager

@dataclass
class AgentConfig:
    """Main configuration, loaded from YAML + env vars."""
    
    # Environment
    display_width: int = 1920
    display_height: int = 1080
    headless: bool = False
    
    # Agent loop
    loop_rate_hz: float = 1.0
    max_actions_per_minute: int = 60
    
    # LLM
    llm_provider: str = "anthropic"  # or "openai"
    llm_model: str = "claude-3-sonnet"
    llm_vision_model: str = "claude-3-sonnet"
    
    # Vision
    ocr_confidence_threshold: float = 0.8
    ui_detection_threshold: float = 0.7
    
    # Observer
    stream_fps: int = 10
    stream_quality: int = 80


class ConfigManager:
    """Manages configuration with runtime update support."""
    
    def __init__(self, config: AgentConfig):
        self._config = config
        self._subscribers: list[Callable] = []
    
    def get(self) -> AgentConfig:
        return self._config
    
    def update(self, updates: dict) -> AgentConfig:
        """Update config at runtime. Notifies subscribers."""
        # Validate updates
        # Apply updates
        # Notify subscribers
        for subscriber in self._subscribers:
            subscriber(self._config)
        return self._config
    
    def subscribe(self, callback: Callable[[AgentConfig], None]) -> None:
        """Subscribe to config changes for hot reload."""
        self._subscribers.append(callback)
```

**Configuration Hierarchy** (highest priority first):
1. Runtime updates via API (7.3)
2. Environment variables (`AGENT_ZERO_*`)
3. Config file (`configs/agent.yaml`)
4. Default values in code

**Acceptance Criteria**:
- [ ] `make install` successfully installs all dependencies
- [ ] `make test` runs (empty) test suite
- [ ] `make lint` passes with no errors
- [ ] Configuration loads from YAML
- [ ] Environment variables override YAML values
- [ ] ConfigManager supports runtime updates
- [ ] Subscribers notified on config change

**Testing Strategy**:
```
TEST: Run `make install && make test && make lint`
PASS: All commands exit 0

TEST: Load config from YAML
PASS: Values match YAML file

TEST: Override config via environment variable
PASS: Env var value takes precedence

TEST: Update config at runtime
PASS: Subscribers receive updated config
```

**Dependencies**: None
**Parallel Work**: Yes - independent

---

#### Feature 0.2: Interface Definitions
**Description**: Define all component interfaces as abstract base classes and Pydantic models.

**Deliverables**:
- `src/interfaces/vision.py` - Vision system interface
- `src/interfaces/actions.py` - Action executor interface (includes InputBackend)
- `src/interfaces/memory.py` - Memory system interface
- `src/interfaces/strategy.py` - Strategy engine interface
- `src/interfaces/environment.py` - Environment manager interface
- `src/interfaces/communication.py` - Streaming/API interface
- `src/models/` - All Pydantic data models

**Interface → Implementation Mapping**:
```
Interface (0.2)                    Implementation (Sprint)
─────────────────────────────────────────────────────────
VisionSystem                    →  2.1-2.4 (combined in 4.1)
├─ ScreenCapture               →  2.1 ScreenshotCapture
├─ OCRSystem                   →  2.2 OCRSystem
├─ UIDetector                  →  2.3 UIDetector
└─ LLMVision                   →  2.4 LLMVision

InputBackend                    →  3.3 PlaywrightInputBackend
ActionExecutor                  →  3.3 ActionExecutor

StatePersistence               →  5.1 SQLitePersistence
EpisodicMemory                 →  5.2 EpisodicMemory
StrategyMemory                 →  5.3 StrategyMemory

StrategyEngine                 →  6.1-6.4
├─ GoalManager                 →  6.1 GoalHierarchy
├─ IncrementalStrategy         →  6.2 IncrementalMetaStrategy
├─ Planner                     →  6.3 PlanningSystem
└─ GameAdapter                 →  6.4 TargetGameAdapter

EnvironmentManager             →  1.3 LocalEnvironmentManager
├─ VirtualDisplay              →  1.3 (embedded)
├─ BrowserRuntime              →  1.2 BrowserRuntime
└─ AuthManager                 →  1.4 RobloxAuth

ScreenStreamer                 →  7.1 ScreenStreaming
LogStreamer                    →  7.2 LogStreaming
ControlAPI                     →  7.3 ControlAPI
```

**Acceptance Criteria**:
- [ ] All interfaces defined with complete method signatures
- [ ] All data models defined with validation
- [ ] Type hints complete and mypy passes
- [ ] Docstrings explain purpose and usage
- [ ] Each interface has a corresponding "Implements" note in later features

**Testing Strategy**:
```
TEST: mypy src/interfaces/ --strict
PASS: No type errors

TEST: Import all interfaces, instantiate mock implementations
PASS: All imports succeed, mocks instantiate

TEST: Verify all interfaces have ABC as base class
PASS: All interfaces are abstract
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
TEST: docker build -t agent-zero .
PASS: Build completes without errors

TEST: docker run agent-zero xeyes &
      Connect via VNC to container:5900
PASS: xeyes window visible in VNC

TEST: docker run agent-zero python --version
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

#### Feature 1.4: Roblox Authentication
**Description**: Handle Roblox login, session management, and authentication persistence.

**Deliverables**:
- `src/environment/auth.py` - Authentication management
- Credential storage (secure, encrypted)
- Session/cookie persistence across restarts
- Login flow automation
- 2FA handling (TOTP support)

**Acceptance Criteria**:
- [ ] Can log into Roblox with username/password
- [ ] Session persists across container restarts
- [ ] Handles 2FA if enabled on account
- [ ] Detects session expiration and re-authenticates
- [ ] Credentials stored securely (not in plaintext)

**Testing Strategy**:
```
TEST: Login with valid credentials
PASS: Successfully authenticated, can access games

TEST: Restart container, verify session persists
PASS: No re-login required

TEST: Expire session manually, verify re-auth
PASS: Automatic re-authentication succeeds

TEST: Login with 2FA-enabled account
PASS: TOTP code accepted, login succeeds

TEST: Verify credentials not in logs or plaintext files
PASS: Security audit passes
```

**Security Notes**:
- Credentials should be provided via environment variables or secrets manager
- Never log credentials or session tokens
- Use Playwright's storage state for cookie persistence

**Dependencies**: 1.2, 1.3
**Parallel Work**: No - requires browser runtime

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

#### Feature 3.3: Action Executor & Input Backend
**Description**: Unified action execution with validation, plus the backend that connects controllers to actual input mechanisms.

**Deliverables**:
- `src/actions/backend.py` - Input backend interface and implementations
- `src/actions/executor.py` - Action execution orchestration
- Pre-action validation
- Post-action verification
- Action queuing and rate limiting

**Input Backend Design**:
```python
# src/actions/backend.py
from abc import ABC, abstractmethod

class InputBackend(ABC):
    """Interface for actual input mechanisms."""
    
    @abstractmethod
    def key_down(self, key: str) -> None: ...
    
    @abstractmethod
    def key_up(self, key: str) -> None: ...
    
    @abstractmethod
    def type_char(self, char: str) -> None: ...
    
    @abstractmethod
    def mouse_move(self, x: int, y: int) -> None: ...
    
    @abstractmethod
    def mouse_down(self, button: str) -> None: ...
    
    @abstractmethod
    def mouse_up(self, button: str) -> None: ...


class PlaywrightInputBackend(InputBackend):
    """Backend that uses Playwright's page.keyboard and page.mouse."""
    
    def __init__(self, page: Page) -> None:
        self._page = page
    
    def key_down(self, key: str) -> None:
        self._page.keyboard.down(key)
    
    def mouse_move(self, x: int, y: int) -> None:
        self._page.mouse.move(x, y)
    # ... etc


class NullInputBackend(InputBackend):
    """Backend that does nothing - for testing."""
    def key_down(self, key: str) -> None: pass
    # ... etc
```

**Controller Integration**:
```python
# Controllers accept optional backend
class KeyboardController:
    def __init__(self, backend: InputBackend | None = None):
        self._backend = backend or NullInputBackend()
    
    def _key_down(self, key: str) -> None:
        self._backend.key_down(key)
```

**Acceptance Criteria**:
- [ ] `InputBackend` interface defined with all input methods
- [ ] `PlaywrightInputBackend` implements interface using page.keyboard/mouse
- [ ] `NullInputBackend` for testing (current stub behavior)
- [ ] Controllers use backend for actual input (not just logging)
- [ ] Executes Action objects from models
- [ ] Validates target exists before click
- [ ] Verifies action had effect (state changed)
- [ ] Rate limits to avoid detection

**Testing Strategy**:
```
TEST: PlaywrightInputBackend.key_down calls page.keyboard.down
PASS: Playwright method called with correct key

TEST: PlaywrightInputBackend.mouse_move calls page.mouse.move  
PASS: Playwright method called with correct coordinates

TEST: KeyboardController with PlaywrightInputBackend types text
PASS: Text appears in browser input field

TEST: MouseController with PlaywrightInputBackend clicks button
PASS: Button is activated in browser

TEST: Execute valid click action through ActionExecutor
PASS: Click performed, state changed

TEST: Execute click on non-existent element
PASS: ActionError raised before attempt

TEST: Rapid-fire 100 actions
PASS: Rate limiting enforces minimum delay
```

**Integration Test Approach**:
Unit tests use `NullInputBackend` (mocked). Integration tests (Sprint 8) use `PlaywrightInputBackend` with a test page containing buttons and input fields.

**Dependencies**: 3.1, 3.2, 0.3, 2.3, 1.2 (for PlaywrightInputBackend)
**Parallel Work**: No - integrates 3.1 and 3.2

---

### Sprint 4: Agent Core

**Objective**: Core orchestration and decision loop.

#### Feature 4.1: Observation Pipeline
**Description**: Capture → Process → Structured State. Combines all vision components into a unified observation.

**Deliverables**:
- `src/core/observation.py` - Observation pipeline
- Vision component orchestration
- Produces Observation objects with full game state

**Vision Component Composition**:
```python
class ObservationPipeline:
    """Orchestrates vision components to produce observations."""
    
    def __init__(
        self,
        capture: ScreenshotCapture,  # 2.1
        ocr: OCRSystem,               # 2.2
        ui_detector: UIDetector,      # 2.3
        llm_vision: LLMVision,        # 2.4
    ):
        self._capture = capture
        self._ocr = ocr
        self._ui_detector = ui_detector
        self._llm_vision = llm_vision
    
    async def observe(self) -> Observation:
        # Step 1: Capture screenshot
        screenshot = await self._capture.capture()
        
        # Step 2: Run OCR and UI detection in PARALLEL
        # (These are independent and can run concurrently)
        ocr_task = asyncio.create_task(self._ocr.extract_text(screenshot))
        ui_task = asyncio.create_task(self._ui_detector.detect(screenshot))
        
        text_regions, ui_elements = await asyncio.gather(ocr_task, ui_task)
        
        # Step 3: Parse resources from OCR results
        resources = self._parse_resources(text_regions)
        
        # Step 4: Use LLM Vision for complex understanding
        # Only if needed (e.g., ambiguous state, new screen)
        if self._needs_llm_analysis(ui_elements, resources):
            game_state = await self._llm_vision.analyze(
                screenshot, 
                ui_elements,  # Provide detected elements for grounding
                text_regions  # Provide OCR results to reduce LLM work
            )
        else:
            # Build game state from OCR + UI detection alone
            game_state = self._build_game_state(resources, ui_elements)
        
        return Observation(
            screenshot=screenshot,
            game_state=game_state,
            ui_elements=ui_elements,
            timestamp=datetime.now()
        )
```

**Component Interaction**:
```
Screenshot (2.1)
     │
     ├──────────────┬─────────────────┐
     ▼              ▼                 ▼
  OCR (2.2)    UI Detect (2.3)   LLM Vision (2.4)
     │              │                 │
     │   [parallel] │    [conditional]│
     ▼              ▼                 ▼
  Resources     Elements          GameState
     │              │                 │
     └──────────────┴─────────────────┘
                    │
                    ▼
              Observation
```

**When to use LLM Vision**:
- First observation (unknown screen)
- Screen transition detected
- Low confidence from OCR/UI detection
- Every N observations (periodic validation)
- NOT every frame (too expensive)

**Acceptance Criteria**:
- [ ] Single call produces complete Observation
- [ ] OCR and UI detection run in parallel
- [ ] LLM Vision called only when needed (conditional)
- [ ] Includes screenshot, game state, UI elements, timestamp
- [ ] Handles individual vision component errors gracefully
- [ ] Performance < 500ms without LLM, < 2s with LLM

**Testing Strategy**:
```
TEST: Call observe() with game running
PASS: Returns valid Observation with all fields

TEST: Verify OCR and UI detection run in parallel
PASS: Total time < max(OCR time, UI time) + overhead

TEST: Measure observation latency over 50 calls (no LLM)
PASS: 95th percentile < 500ms

TEST: Measure observation latency with LLM
PASS: 95th percentile < 2 seconds

TEST: Inject OCR error, verify graceful handling
PASS: Observation returned with partial data, error logged

TEST: Verify LLM not called on every observation
PASS: LLM call count << observation count
```

**Dependencies**: 2.1, 2.2, 2.3, 2.4
**Parallel Work**: No - integrates vision components

---

#### Feature 4.2: Decision Engine
**Description**: ReAct-style reasoning and action selection using LLM.

**Deliverables**:
- `src/core/decision.py` - Decision making
- `src/core/prompts/` - Prompt templates for decision making
- ReAct prompting implementation
- Chain-of-thought reasoning
- Action selection with confidence

**LLM Integration**:
```python
# Shares LLM client with Vision (2.4) to reduce complexity
# Uses same Anthropic/OpenAI client, different prompts

class DecisionEngine:
    def __init__(self, llm_client: LLMClient, strategy_context: StrategyContext):
        self._llm = llm_client  # Same client as LLMVision
        self._prompts = DecisionPrompts()
    
    def decide(self, observation: Observation, memory: MemoryContext) -> Decision:
        # Build ReAct prompt with:
        # - Current game state (from observation)
        # - Recent actions and outcomes (from memory)
        # - Available strategies
        # - Goal hierarchy
        prompt = self._prompts.build_react_prompt(observation, memory)
        
        response = self._llm.complete(prompt)
        return self._parse_decision(response)
```

**ReAct Prompt Structure**:
```
OBSERVATION: [Game state from 4.1]
THOUGHT: [LLM reasons about current situation]
ACTION: [Selected action with target]
EXPECTED: [What should happen next]
```

**Acceptance Criteria**:
- [ ] Takes Observation, returns Decision
- [ ] Decision includes reasoning trace (THOUGHT)
- [ ] Confidence scores are calibrated
- [ ] Handles ambiguous states gracefully
- [ ] Shares LLM client with Vision module
- [ ] Prompts are versioned and configurable

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

TEST: LLM client is same instance as Vision module
PASS: No duplicate client initialization
```

**Cost Considerations**:
- Decision calls happen every loop iteration (potentially 1/second)
- Use smaller/faster model for decisions vs. vision analysis
- Cache decisions for identical game states
- Consider local LLM fallback for simple decisions

**Dependencies**: 0.3, 4.1, 2.4 (shares LLM client)
**Parallel Work**: Yes - uses Observation interface

---

#### Feature 4.3: Main Loop & Metrics
**Description**: Observe → Decide → Act → Repeat. Collects and exposes metrics for monitoring.

**Deliverables**:
- `src/core/loop.py` - Main agent loop
- `src/core/metrics.py` - Metrics collection and storage
- Configurable loop rate
- Error recovery
- Graceful shutdown
- Pause/resume support

**Metrics Collection**:
```python
@dataclass
class AgentMetrics:
    """Metrics collected by the main loop."""
    
    # Timing
    loop_count: int = 0
    loop_rate_hz: float = 0.0
    avg_observation_time_ms: float = 0.0
    avg_decision_time_ms: float = 0.0
    avg_action_time_ms: float = 0.0
    
    # LLM usage
    llm_calls_total: int = 0
    llm_calls_vision: int = 0
    llm_calls_decision: int = 0
    llm_tokens_total: int = 0
    llm_cost_usd: float = 0.0
    
    # Actions
    actions_total: int = 0
    actions_successful: int = 0
    actions_failed: int = 0
    actions_per_minute: float = 0.0
    
    # Errors
    errors_total: int = 0
    errors_recovered: int = 0
    errors_by_type: dict[str, int] = field(default_factory=dict)
    
    # Game progress
    current_goal: str = ""
    goal_progress_percent: float = 0.0
    resources_snapshot: dict[str, float] = field(default_factory=dict)
    
    # Uptime
    started_at: datetime | None = None
    uptime_seconds: float = 0.0


class MetricsCollector:
    """Collects metrics during loop execution."""
    
    def record_loop_iteration(self, duration_ms: float) -> None: ...
    def record_observation(self, duration_ms: float) -> None: ...
    def record_decision(self, duration_ms: float, llm_used: bool) -> None: ...
    def record_action(self, success: bool, duration_ms: float) -> None: ...
    def record_error(self, error_type: str, recovered: bool) -> None: ...
    def record_llm_call(self, component: str, tokens: int, cost: float) -> None: ...
    
    def get_metrics(self) -> AgentMetrics: ...
```

**Loop Structure**:
```python
class AgentLoop:
    def __init__(
        self,
        observation_pipeline: ObservationPipeline,
        decision_engine: DecisionEngine,
        action_executor: ActionExecutor,
        metrics: MetricsCollector,
    ):
        self._state = "stopped"  # stopped, running, paused
        
    async def run(self):
        self._state = "running"
        while self._state == "running":
            try:
                # Observe
                with self._metrics.time("observation"):
                    observation = await self._observation_pipeline.observe()
                
                # Decide
                with self._metrics.time("decision"):
                    decision = await self._decision_engine.decide(observation)
                
                # Act
                with self._metrics.time("action"):
                    result = await self._action_executor.execute(decision.action)
                    self._metrics.record_action(result.success, result.duration_ms)
                
                # Rate limit
                await self._rate_limiter.wait()
                
            except RecoverableError as e:
                self._metrics.record_error(type(e).__name__, recovered=True)
                logger.warning(f"Recovered from error: {e}")
                
            except FatalError as e:
                self._metrics.record_error(type(e).__name__, recovered=False)
                self._state = "error"
                raise
    
    def pause(self) -> None:
        self._state = "paused"
    
    def resume(self) -> None:
        if self._state == "paused":
            self._state = "running"
    
    def stop(self) -> None:
        self._state = "stopped"
```

**Acceptance Criteria**:
- [ ] Runs continuously until stopped
- [ ] Each iteration: observe, decide, act
- [ ] Collects timing metrics for each phase
- [ ] Tracks LLM usage and cost
- [ ] Tracks action success/failure rates
- [ ] Recovers from transient errors
- [ ] Logs each iteration for debugging
- [ ] Clean shutdown on signal (SIGTERM/SIGINT)
- [ ] Supports pause/resume
- [ ] Metrics accessible via `get_metrics()`

**Testing Strategy**:
```
TEST: Run loop for 10 iterations, verify progression
PASS: All 10 iterations complete, game state progresses

TEST: Inject error on iteration 5
PASS: Error logged, loop continues, error counted in metrics

TEST: Send SIGTERM during loop
PASS: Clean shutdown, no orphan processes

TEST: Verify logs contain observation, decision, action
PASS: All components logged each iteration

TEST: Get metrics after 100 iterations
PASS: All metric fields populated with reasonable values

TEST: Pause loop, verify no iterations
PASS: Loop count stays constant while paused

TEST: Resume loop, verify iterations continue
PASS: Loop count increases after resume
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

**Implements**: `src/interfaces/memory.py` (StatePersistence interface)

**Acceptance Criteria**:
- [ ] Implements StatePersistence interface from 0.2
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

TEST: Verify implements StatePersistence interface
PASS: isinstance(persistence, StatePersistence) == True
```

**Dependencies**: 0.3, 0.2 (interface)
**Parallel Work**: Yes - independent module

---

#### Feature 5.2: Episodic Memory
**Description**: Remember what happened and what worked.

**Deliverables**:
- `src/memory/episodic.py` - Episode tracking
- Action-outcome pairs
- Success/failure annotations
- Similarity search for relevant episodes

**Implements**: `src/interfaces/memory.py` (EpisodicMemory interface)

**Acceptance Criteria**:
- [ ] Implements EpisodicMemory interface from 0.2
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

TEST: Verify implements EpisodicMemory interface
PASS: isinstance(memory, EpisodicMemory) == True
```

**Dependencies**: 5.1, 0.3, 0.2 (interface)
**Parallel Work**: Yes - parallel with 5.3

---

#### Feature 5.3: Strategy Memory
**Description**: Remember high-level strategies and patterns.

**Deliverables**:
- `src/memory/strategy.py` - Strategy tracking
- Strategy effectiveness scores
- Pattern recognition

**Implements**: `src/interfaces/memory.py` (StrategyMemory interface)

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

**Dependencies**: 5.1, 0.3 (NOT 5.2 - these are parallel)
**Parallel Work**: Yes - parallel with 5.2, both depend only on 5.1

**Note**: 5.2 (Episodic) and 5.3 (Strategy) are independent memory systems. 
Strategy memory tracks named strategies and their effectiveness.
Episodic memory tracks individual action→outcome pairs.
They can be developed in parallel and may optionally share data later.

---

### Sprint 6: Strategy

**Objective**: Game-specific intelligence and optimization.

#### Feature 6.0: Target Game Selection & Analysis
**Description**: Select, analyze, and document the target Roblox incremental game before building game-specific components.

**Deliverables**:
- `docs/target_game.md` - Comprehensive game documentation
- `tests/fixtures/game/` - Real screenshots from target game
- `configs/game_config.yaml` - Game-specific configuration
- Game mechanics analysis
- UI element catalog
- Win condition definition

**Selection Criteria**:
1. **Publicly available** (not private/invite-only)
2. **Pure incremental** (click/upgrade loop, minimal action requirements)
3. **Reasonable scope** (completable in hours, not months)
4. **Stable** (not frequently updated, breaking automation)
5. **Clear UI** (buttons and resources easily detectable)

**Candidates**:
- Clicker Simulator
- Tapping Legends
- Clicking Legends
- Mining Simulator (incremental aspects)

**Analysis Process**:
```
1. Play each candidate game manually for 30 minutes
2. Document:
   - Core game loop (what actions repeat?)
   - Resource types and display format
   - Upgrade system and UI
   - Prestige/reset mechanics
   - Win condition or "completion" state
3. Capture 50+ screenshots covering all UI states
4. Annotate UI elements for training/testing
5. Select game with best automation potential
```

**Deliverable: docs/target_game.md**:
```markdown
# Target Game: [Selected Game Name]

## Overview
- Game URL: [roblox.com/games/...]
- Genre: Incremental/Clicker
- Estimated completion time: X hours

## Core Loop
1. Click to earn [resource]
2. Buy upgrades to increase [resource] per click
3. Buy auto-clickers
4. Prestige at [threshold] to gain multipliers
5. Repeat until [win condition]

## UI Elements
| Element | Location | Detection Method |
|---------|----------|------------------|
| Gold counter | Top-left | OCR, format "1.5M" |
| Upgrade button | Right panel | Template match |
| ...

## Screenshots
- `fixtures/game/main_screen.png` - Default game view
- `fixtures/game/upgrade_menu.png` - Upgrade panel open
- `fixtures/game/prestige_dialog.png` - Prestige confirmation
- ...

## Win Condition
[Description of what "completing" the game means]
```

**Acceptance Criteria**:
- [ ] At least 3 candidate games evaluated
- [ ] Target game selected with documented rationale
- [ ] 50+ screenshots captured covering all UI states
- [ ] All UI elements cataloged with detection strategy
- [ ] Win condition clearly defined
- [ ] Game config file created

**Testing Strategy**:
```
TEST: Load game screenshots, verify UI elements detectable
PASS: OCR extracts resource values, UI detection finds buttons

TEST: Game config loads without errors
PASS: All required fields present and valid

TEST: Fixtures directory contains required screenshots
PASS: main_screen, upgrade_menu, prestige screens present
```

**Dependencies**: 1.4 (need to log into Roblox), 2.1-2.4 (use vision to verify detectability)
**Parallel Work**: No - must complete before 6.1-6.4
**Note**: This is a research/documentation task, not pure coding.

---

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

**Dependencies**: 6.0 (game analysis), 6.1, 6.2, 6.3, 2.3
**Parallel Work**: No - requires 6.0 game selection and other strategy components

---

### Sprint 7: Observer

**Objective**: User visibility and control interface.

**Tech Stack Decision**:
- **Backend**: FastAPI (async, WebSocket support, OpenAPI docs)
- **Frontend**: React SPA with TypeScript
- **Communication**: WebSocket for streaming, REST for control
- **Deployment**: Served from same container as agent

#### Feature 7.1: Screen Streaming
**Description**: Stream agent's screen to user via WebSocket.

**Deliverables**:
- `src/observer/streaming.py` - Screen streaming server
- `src/observer/server.py` - FastAPI application setup
- WebSocket endpoint for frame streaming
- JPEG compression with configurable quality
- Frame rate limiting

**Implements**: `src/interfaces/communication.py` (ScreenStreamer interface)

**API Design**:
```python
# WebSocket endpoint: ws://host:8000/ws/screen
# Sends: Binary JPEG frames
# Receives: Quality/FPS configuration messages

@app.websocket("/ws/screen")
async def screen_stream(websocket: WebSocket):
    await websocket.accept()
    while True:
        frame = await capture_service.get_latest_frame()
        jpeg = compress_frame(frame, quality=settings.stream_quality)
        await websocket.send_bytes(jpeg)
        await asyncio.sleep(1 / settings.stream_fps)
```

**Acceptance Criteria**:
- [ ] FastAPI WebSocket endpoint serves frames
- [ ] Streams screen at 10+ FPS
- [ ] Latency < 500ms
- [ ] Quality adjustable via WebSocket message
- [ ] Handles reconnection gracefully

**Testing Strategy**:
```
TEST: Connect WebSocket client, verify frames received
PASS: Frames arrive at 10+ FPS

TEST: Measure end-to-end latency
PASS: 95th percentile < 500ms

TEST: Disconnect and reconnect
PASS: Streaming resumes within 2 seconds

TEST: Send quality adjustment message
PASS: Subsequent frames reflect new quality
```

**Dependencies**: 2.1
**Parallel Work**: Yes - uses Vision interface

---

#### Feature 7.2: Log Streaming
**Description**: Stream agent reasoning to user via WebSocket.

**Deliverables**:
- `src/observer/logs.py` - Log streaming and formatting
- WebSocket endpoint for log streaming
- Structured JSON log format
- Log levels (debug, info, decision, error)
- Sensitive data redaction

**Implements**: `src/interfaces/communication.py` (LogStreamer interface)

**API Design**:
```python
# WebSocket endpoint: ws://host:8000/ws/logs
# Sends: JSON log entries
# Receives: Filter configuration (log level, component)

@app.websocket("/ws/logs")
async def log_stream(websocket: WebSocket):
    await websocket.accept()
    async for log_entry in log_queue.subscribe():
        if passes_filter(log_entry, current_filter):
            await websocket.send_json(log_entry.dict())

# Log entry format:
{
    "timestamp": "2024-01-15T10:30:00.123Z",
    "level": "decision",
    "component": "decision_engine",
    "message": "Selected action: click upgrade button",
    "data": {
        "reasoning": "Resource count high enough for upgrade",
        "confidence": 0.85,
        "action": {"type": "click", "target": [450, 320]}
    }
}
```

**Acceptance Criteria**:
- [ ] WebSocket endpoint streams logs
- [ ] Logs stream in real-time (<1s latency)
- [ ] Decision reasoning visible in structured format
- [ ] Log levels filterable via WebSocket message
- [ ] No sensitive data leaked (API keys, credentials redacted)

**Testing Strategy**:
```
TEST: Trigger agent decision, verify log appears
PASS: Decision log received within 1 second

TEST: Send filter message for "decision" level only
PASS: Only decision-level logs received

TEST: Verify no API keys in logs
PASS: Sensitive data redacted (regex check)

TEST: Log entry contains required fields
PASS: timestamp, level, component, message all present
```

**Dependencies**: 4.3
**Parallel Work**: Yes - parallel with 7.1

---

#### Feature 7.3: Control API
**Description**: REST API for user to control agent (start, stop, configure).

**Deliverables**:
- `src/observer/control.py` - Control endpoints
- `src/observer/schemas.py` - Pydantic request/response models
- REST API for commands
- Start/stop/pause controls
- Configuration updates with validation
- Status and metrics endpoints

**Implements**: `src/interfaces/communication.py` (ControlAPI interface)

**API Design**:
```python
# FastAPI REST endpoints

@app.get("/api/status")
async def get_status() -> AgentStatus:
    """Get current agent status, uptime, and health."""
    return AgentStatus(
        state="running",  # running, paused, stopped, error
        uptime_seconds=3600,
        current_goal="Reach level 10",
        actions_performed=1250,
        errors_count=2
    )

@app.post("/api/start")
async def start_agent() -> ActionResponse:
    """Start the agent loop."""

@app.post("/api/stop")
async def stop_agent() -> ActionResponse:
    """Stop the agent gracefully."""

@app.post("/api/pause")
async def pause_agent() -> ActionResponse:
    """Pause agent (stops actions but maintains state)."""

@app.post("/api/resume")
async def resume_agent() -> ActionResponse:
    """Resume paused agent."""

@app.get("/api/config")
async def get_config() -> AgentConfig:
    """Get current configuration."""

@app.patch("/api/config")
async def update_config(updates: ConfigUpdate) -> AgentConfig:
    """Update configuration at runtime."""
    # Validates updates, applies to running agent
    # Some settings require restart (returns restart_required=True)

@app.get("/api/metrics")
async def get_metrics() -> AgentMetrics:
    """Get performance metrics."""
    return AgentMetrics(
        loop_rate_hz=1.2,
        avg_decision_time_ms=450,
        llm_calls_total=500,
        llm_cost_usd=1.25,
        actions_per_minute=15
    )
```

**Acceptance Criteria**:
- [ ] All endpoints documented via OpenAPI (auto-generated)
- [ ] Start agent via POST /api/start
- [ ] Stop agent cleanly via POST /api/stop
- [ ] Pause/resume agent
- [ ] Update configuration at runtime via PATCH /api/config
- [ ] Status endpoint returns current state
- [ ] Metrics endpoint returns performance data

**Testing Strategy**:
```
TEST: POST /api/start, verify agent starts
PASS: Agent running, GET /api/status confirms

TEST: POST /api/stop, verify clean shutdown
PASS: Agent stopped, no orphan processes

TEST: PATCH /api/config with new setting
PASS: Agent behavior reflects new config

TEST: GET /api/metrics returns valid data
PASS: All metric fields present and reasonable

TEST: OpenAPI docs accessible at /docs
PASS: Swagger UI loads with all endpoints
```

**Dependencies**: 4.3
**Parallel Work**: Yes - parallel with 7.1, 7.2

---

#### Feature 7.4: Web Dashboard
**Description**: React SPA for observation and control.

**Deliverables**:
- `frontend/` - React application (separate directory)
- `frontend/src/components/ScreenViewer.tsx` - Live screen component
- `frontend/src/components/LogViewer.tsx` - Scrolling log component  
- `frontend/src/components/MetricsPanel.tsx` - Metrics display
- `frontend/src/components/ControlPanel.tsx` - Start/stop/config controls
- `frontend/src/hooks/useWebSocket.ts` - WebSocket connection management
- Static files served by FastAPI in production

**Tech Stack**:
- React 18 with TypeScript
- Tailwind CSS for styling
- Zustand for state management
- Vite for build tooling

**Component Layout**:
```
┌─────────────────────────────────────────────────────────┐
│  Agent Zero Dashboard                    [▶ Start] [⏹]  │
├─────────────────────────────────┬───────────────────────┤
│                                 │  Metrics              │
│   Live Screen                   │  ├─ Loop rate: 1.2 Hz │
│   (WebSocket stream)            │  ├─ Actions: 1,250    │
│                                 │  ├─ LLM cost: $1.25   │
│   [1920x1080 viewport]          │  └─ Uptime: 1h 30m    │
│                                 ├───────────────────────┤
│                                 │  Current Goal         │
│                                 │  "Reach level 10"     │
│                                 │  Progress: ████░░ 67% │
├─────────────────────────────────┴───────────────────────┤
│  Logs                                        [Filter ▼] │
│  10:30:01 [decision] Click upgrade button (conf: 0.85)  │
│  10:30:00 [vision] Detected: Gold=1.5M, Level=7         │
│  10:29:58 [action] Clicked at (450, 320)                │
│  ...                                                    │
└─────────────────────────────────────────────────────────┘
```

**Acceptance Criteria**:
- [ ] React SPA with TypeScript
- [ ] WebSocket connection to screen stream
- [ ] WebSocket connection to log stream
- [ ] REST calls to control API
- [ ] Shows live screen (10+ FPS)
- [ ] Shows scrolling logs with level filtering
- [ ] Control buttons (start/stop/pause) work
- [ ] Metrics update in real-time
- [ ] Works in Chrome, Firefox, Safari
- [ ] Responsive (works on tablet)

**Testing Strategy**:
```
TEST: Load dashboard, verify components render
PASS: Screen, logs, controls, metrics all visible

TEST: Click start button, verify effect
PASS: Agent starts, status updates to "running"

TEST: Verify screen updates at 10+ FPS
PASS: Frame counter shows consistent rate

TEST: Filter logs by level
PASS: Only matching log levels displayed

TEST: Run for 5 minutes, verify stability
PASS: No memory leaks (Chrome DevTools check)

TEST: Disconnect network, reconnect
PASS: WebSocket reconnects, streaming resumes
```

**Build & Deployment**:
```bash
# Development
cd frontend && npm run dev  # Vite dev server with hot reload

# Production  
cd frontend && npm run build  # Outputs to frontend/dist/
# FastAPI serves static files from frontend/dist/
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

Sprint 1: [1.1] ──> [1.2] ──> [1.3] ──> [1.4]
                              │
Sprint 2: [2.1] ──┬──> [2.2]  │
                 ├──> [2.3]  │
                 └──> [2.4]  │
                              │
Sprint 3: [3.1] ──┐           │
                 ├──> [3.3] <─┘ (needs 1.2 for PlaywrightInputBackend)
          [3.2] ──┘

Sprint 4: [4.1] ──> [4.2] ──> [4.3]
          (combines 2.1-2.4)

Sprint 5: [5.1] ──┬──> [5.2]
                 └──> [5.3]  (parallel, both depend on 5.1 only)

Sprint 6: [6.0] ──> [6.1] ──┬──> [6.2] ──┐
          (game selection)  └──> [6.3] ──┼──> [6.4]
                                         │
Sprint 7: [7.1] ──┐                      │
          [7.2] ──┼──> [7.4]             │
          [7.3] ──┘                      │
                                         │
Sprint 8: [8.1] <────────────────────────┘
             │    (requires ALL previous sprints)
             v
          [8.2]

Sprint 9: [9.1]
          [9.2]  (all parallel, depend only on 8.1)
          [9.3]
```

**Key Dependency Notes**:
- **1.4 (Auth)**: Requires browser (1.2) and environment manager (1.3)
- **3.3 (Action Executor)**: Requires 1.2 for `PlaywrightInputBackend`
- **4.2 (Decision)**: Shares LLM client with 2.4
- **5.2 and 5.3**: Both depend on 5.1, NOT on each other (parallel)
- **6.0 (Game Selection)**: Must complete before any 6.x features
- **6.4 (Game Adapter)**: Depends on 6.0 analysis and 6.1-6.3 systems

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
       /    \     - Full agent playing real game
      /──────\    
     /        \   Integration Tests (Sprint 8)
    /          \  - Components with real browser
   /────────────\ 
  /              \ Unit Tests (Each Sprint)
 /                \ - Individual functions, mocked deps
/──────────────────\ - Run without Docker/browser
```

### Test Categories

**1. Unit Tests (Each Feature)**
- Run without Docker or browser
- Mock all external dependencies (LLM APIs, Playwright, filesystem)
- Fast (<1 second per test)
- Example: `KeyboardController` with `NullInputBackend`

**2. Integration Tests (Sprint 8)**
- Run inside Docker container with virtual display
- Use real Playwright with test pages
- Test component combinations
- Example: `KeyboardController` + `PlaywrightInputBackend` + test HTML page

**3. E2E Tests (Sprint 8)**
- Run full agent against real Roblox game
- Measure actual game progress
- Long-running (30+ minutes)

### Integration Test Infrastructure

```
tests/
├── unit/                    # Unit tests (mock everything)
│   ├── test_keyboard.py
│   ├── test_mouse.py
│   └── ...
├── integration/             # Integration tests (real browser)
│   ├── fixtures/
│   │   └── test_page.html   # HTML page with buttons, inputs
│   ├── test_input_backend.py
│   ├── test_action_executor.py
│   └── ...
├── e2e/                     # End-to-end tests
│   ├── test_game_progression.py
│   └── ...
└── fixtures/
    ├── ocr/                 # Test images for OCR
    ├── ui/                  # Test images for UI detection
    └── game/                # Real game screenshots (6.0)
```

**Test HTML Page** (for integration tests):
```html
<!-- tests/integration/fixtures/test_page.html -->
<html>
<body>
  <input id="text-input" type="text" />
  <button id="test-button" onclick="this.classList.add('clicked')">
    Click Me
  </button>
  <div id="output"></div>
  <script>
    document.getElementById('text-input').addEventListener('input', (e) => {
      document.getElementById('output').textContent = e.target.value;
    });
  </script>
</body>
</html>
```

**Integration Test Example**:
```python
# tests/integration/test_input_backend.py
@pytest.mark.integration
async def test_keyboard_types_into_input():
    """Verify keyboard actually types into browser input."""
    async with BrowserRuntime() as browser:
        await browser.navigate("file://tests/integration/fixtures/test_page.html")
        
        backend = PlaywrightInputBackend(browser.page)
        keyboard = KeyboardController(backend=backend)
        
        # Click input field
        await browser.page.click("#text-input")
        
        # Type text
        keyboard.type_text("Hello World")
        
        # Verify text appeared
        output = await browser.page.text_content("#output")
        assert output == "Hello World"
```

### Test Requirements

1. **Unit Tests**: Every feature has unit tests BEFORE implementation
2. **Test Fixtures**: Shared fixtures in `tests/fixtures/`
3. **Mocking**: External services (LLM APIs, Playwright) mocked in unit tests
4. **Coverage**: Minimum 80% code coverage (unit tests)
5. **Integration Tests**: Added in Sprint 8, use real browser
6. **Performance Tests**: Latency benchmarks in integration tests

### Test-First Workflow

```
1. Read feature specification
2. Write test file with all test cases (they will fail)
3. Implement feature until tests pass
4. Refactor if needed (tests still pass)
5. PR includes tests + implementation
```

### Running Tests

```bash
# Unit tests only (fast, no Docker)
make test

# Integration tests (requires Docker)
make test-integration

# All tests
make test-all

# E2E tests (requires Roblox account)
make test-e2e
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

## Revision History

| Date | Changes |
|------|---------|
| Initial | Original plan created |
| Rev 1 | Added Feature 1.4 (Roblox Authentication) |
| Rev 1 | Expanded Feature 3.3 with InputBackend interface |
| Rev 1 | Added Feature 6.0 (Target Game Selection) |
| Rev 1 | Clarified vision composition in 4.1 |
| Rev 1 | Added metrics collection to 4.3 |
| Rev 1 | Specified FastAPI + React tech stack for Sprint 7 |
| Rev 1 | Fixed 5.3 dependency (parallel with 5.2, not dependent) |
| Rev 1 | Clarified LLM integration in 4.2 |
| Rev 1 | Added interface→implementation mapping |
| Rev 1 | Added configuration propagation details |
| Rev 1 | Added integration test infrastructure |
| Rev 1 | Updated architecture diagram with Input Layer |

---

## Next Steps

1. **Review revised plan** - Ensure all gaps are addressed
2. **Continue Sprint 3** - Complete 3.3 Action Executor with InputBackend
3. **Begin Sprint 4** - Start Observation Pipeline (4.1)
4. **Begin Sprint 5** - Start Game State Persistence (5.1) in parallel
