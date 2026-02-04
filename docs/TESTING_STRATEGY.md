# Testing Strategy

> **Purpose**: Define testing approach BEFORE building. Every feature knows exactly how it will be tested.

---

## Testing Principles

1. **Test First**: Write tests before implementation
2. **Specific Assertions**: Tests check specific, measurable outcomes
3. **Isolated Tests**: Unit tests don't depend on external services
4. **Reproducible**: Same test, same result, every time
5. **Fast Feedback**: Unit tests run in < 1 second each

---

## Test Categories

### Unit Tests
- Test individual functions/classes
- Mock all dependencies
- Run in milliseconds
- Location: `tests/unit/`

### Integration Tests
- Test component combinations
- May use real local resources (SQLite, files)
- Mock external APIs (LLM)
- Run in seconds
- Location: `tests/integration/`

### End-to-End Tests
- Test full system operation
- Use containerized environment
- Run in minutes
- Location: `tests/e2e/`

### Performance Tests
- Measure latency, throughput, resource usage
- Run in controlled environment
- Location: `tests/performance/`

---

## Mocking Strategy

### LLM APIs (Claude, GPT-4V)

```python
# tests/mocks/llm.py

class MockLLMClient:
    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.call_log = []
    
    def complete(self, prompt: str) -> str:
        self.call_log.append(prompt)
        # Return pre-defined response based on prompt pattern
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return response
        return self.responses.get("default", "{}")
```

Usage:
```python
def test_decision_making():
    mock_llm = MockLLMClient({
        "decide next action": '{"action": "click", "target": "upgrade_button"}',
    })
    engine = DecisionEngine(llm=mock_llm)
    decision = engine.decide(mock_observation)
    assert decision.action.type == "click"
```

### Screenshot Capture

```python
# tests/mocks/capture.py

class MockScreenCapture:
    def __init__(self, fixture_path: str):
        self.frames = self._load_frames(fixture_path)
        self.frame_index = 0
    
    def capture(self) -> bytes:
        frame = self.frames[self.frame_index % len(self.frames)]
        self.frame_index += 1
        return frame
```

### Environment Manager

```python
# tests/mocks/environment.py

class MockEnvironment:
    def __init__(self):
        self.running = False
        self.actions_executed = []
    
    def start(self):
        self.running = True
    
    def stop(self):
        self.running = False
    
    def execute_action(self, action):
        self.actions_executed.append(action)
        return {"success": True}
```

---

## Test Fixtures

### Screenshot Fixtures

Location: `tests/fixtures/screenshots/`

| Fixture | Description | Used By |
|---------|-------------|---------|
| `main_menu.png` | Game main menu | Vision, UI detection |
| `gameplay_early.png` | Early game state | Vision, Strategy |
| `gameplay_mid.png` | Mid game with upgrades | Vision, Strategy |
| `upgrade_screen.png` | Upgrade selection screen | Vision, UI detection |
| `error_disconnect.png` | Disconnection error | Error handling |
| `loading.png` | Loading screen | State detection |

### OCR Fixtures

Location: `tests/fixtures/ocr/`

```json
// tests/fixtures/ocr/numbers.json
{
  "test_cases": [
    {"input": "1.5K", "expected": 1500},
    {"input": "2.3M", "expected": 2300000},
    {"input": "1.2B", "expected": 1200000000},
    {"input": "999", "expected": 999},
    {"input": "1,234", "expected": 1234},
    {"input": "1.23e6", "expected": 1230000}
  ]
}
```

### Game State Fixtures

Location: `tests/fixtures/game_states/`

```json
// tests/fixtures/game_states/early_game.json
{
  "resources": {
    "coins": 150,
    "gems": 0,
    "energy": 100
  },
  "upgrades": [
    {"id": "click_power", "level": 1, "cost": 100},
    {"id": "auto_click", "level": 0, "cost": 500}
  ],
  "current_screen": "main_gameplay",
  "playtime_seconds": 120
}
```

---

## Component Test Specifications

### Feature 0.1: Project Scaffolding

```python
# tests/test_scaffolding.py

def test_dependencies_install():
    """make install completes without error."""
    result = subprocess.run(["make", "install"], capture_output=True)
    assert result.returncode == 0

def test_lint_passes():
    """make lint passes on clean project."""
    result = subprocess.run(["make", "lint"], capture_output=True)
    assert result.returncode == 0

def test_config_loads_yaml():
    """Configuration loads from YAML file."""
    config = load_config("configs/default.yaml")
    assert config.agent.loop_interval > 0

def test_config_env_override():
    """Environment variables override YAML config."""
    os.environ["AGENTZERO_AGENT_LOOP_INTERVAL"] = "5.0"
    config = load_config("configs/default.yaml")
    assert config.agent.loop_interval == 5.0
```

### Feature 0.2: Interface Definitions

```python
# tests/test_interfaces.py

def test_vision_interface_complete():
    """VisionSystem interface has all required methods."""
    required = ["capture", "extract_state", "observe"]
    for method in required:
        assert hasattr(VisionSystem, method)
        assert callable(getattr(VisionSystem, method))

def test_mock_vision_implements_interface():
    """Mock implementation satisfies interface."""
    mock = MockVisionSystem()
    assert isinstance(mock, VisionSystem)
    # All methods callable
    mock.capture()
    mock.extract_state(b"fake")
    mock.observe()
```

### Feature 0.3: Shared Data Models

```python
# tests/test_models.py

def test_game_state_validation():
    """GameState validates required fields."""
    with pytest.raises(ValidationError):
        GameState()  # Missing required fields
    
    state = GameState(
        resources={"coins": 100},
        upgrades=[],
        current_screen="main",
        ui_elements=[],
        timestamp=datetime.now()
    )
    assert state.resources["coins"] == 100

def test_game_state_serialization():
    """GameState round-trips through JSON."""
    state = GameState(...)
    json_str = state.model_dump_json()
    restored = GameState.model_validate_json(json_str)
    assert state == restored

def test_action_types():
    """All action types are valid."""
    for action_type in ActionType:
        action = Action(type=action_type, target=None, parameters={})
        assert action.type == action_type
```

### Feature 1.1: Container Definition

```python
# tests/test_container.py

def test_container_builds():
    """Docker container builds successfully."""
    result = subprocess.run(
        ["docker", "build", "-t", "agent-zero-test", "."],
        capture_output=True,
        timeout=300
    )
    assert result.returncode == 0

def test_virtual_display_starts():
    """Virtual display (Xvfb) starts in container."""
    result = subprocess.run(
        ["docker", "run", "--rm", "agent-zero-test", 
         "xdpyinfo", "-display", ":99"],
        capture_output=True
    )
    assert result.returncode == 0
    assert b"dimensions:" in result.stdout

def test_graphical_app_runs():
    """Graphical application runs in container."""
    # xeyes is a simple X11 app
    result = subprocess.run(
        ["docker", "run", "--rm", "-d", "agent-zero-test",
         "timeout", "5", "xeyes"],
        capture_output=True
    )
    assert result.returncode == 0
```

### Feature 2.1: Screenshot Capture

```python
# tests/vision/test_capture.py

def test_capture_returns_bytes():
    """capture() returns PNG image bytes."""
    capture = ScreenCapture(display=":99")
    data = capture.capture()
    assert isinstance(data, bytes)
    assert data[:8] == b'\x89PNG\r\n\x1a\n'  # PNG magic bytes

def test_capture_framerate():
    """Capture achieves 10+ FPS."""
    capture = ScreenCapture(display=":99")
    start = time.time()
    for _ in range(100):
        capture.capture()
    elapsed = time.time() - start
    fps = 100 / elapsed
    assert fps >= 10

def test_capture_buffer():
    """Buffer stores last N frames."""
    capture = ScreenCapture(display=":99", buffer_size=10)
    for _ in range(20):
        capture.capture()
    assert len(capture.buffer) == 10
    # Verify buffer contains most recent frames
    assert capture.buffer[-1] == capture.last_frame
```

### Feature 2.2: OCR System

```python
# tests/vision/test_ocr.py

@pytest.fixture
def ocr_engine():
    return OCREngine()

def test_ocr_basic_text(ocr_engine):
    """OCR extracts visible text."""
    image = load_fixture("screenshots/main_menu.png")
    result = ocr_engine.extract_text(image)
    assert "Play" in result.text  # Known text in fixture

def test_ocr_number_parsing(ocr_engine):
    """OCR parses abbreviated numbers."""
    test_cases = load_fixture("ocr/numbers.json")["test_cases"]
    for case in test_cases:
        result = ocr_engine.parse_number(case["input"])
        assert result == case["expected"], f"Failed: {case['input']}"

def test_ocr_region_extraction(ocr_engine):
    """OCR extracts text from specific region."""
    image = load_fixture("screenshots/gameplay_early.png")
    region = (100, 50, 200, 80)  # Resource counter area
    result = ocr_engine.extract_text(image, region=region)
    # Should only get text from that region
    assert len(result.text) < 50  # Not entire screen

def test_ocr_bounding_boxes(ocr_engine):
    """OCR returns bounding boxes for text."""
    image = load_fixture("screenshots/main_menu.png")
    result = ocr_engine.extract_text(image, with_boxes=True)
    assert len(result.boxes) > 0
    for box in result.boxes:
        assert all(k in box for k in ["x", "y", "width", "height", "text"])
```

### Feature 2.3: UI Element Detection

```python
# tests/vision/test_ui_detection.py

@pytest.fixture
def detector():
    return UIElementDetector()

def test_button_detection(detector):
    """Detects buttons in screenshot."""
    image = load_fixture("screenshots/main_menu.png")
    elements = detector.detect(image)
    buttons = [e for e in elements if e.type == "button"]
    assert len(buttons) >= 3  # Known button count in fixture

def test_detection_confidence(detector):
    """All detections have confidence scores."""
    image = load_fixture("screenshots/gameplay_early.png")
    elements = detector.detect(image)
    for element in elements:
        assert 0 <= element.confidence <= 1

def test_detection_bounding_boxes(detector):
    """Bounding boxes are valid."""
    image = load_fixture("screenshots/gameplay_early.png")
    elements = detector.detect(image)
    for element in elements:
        assert element.x >= 0
        assert element.y >= 0
        assert element.width > 0
        assert element.height > 0

def test_detection_iou(detector):
    """Detection accuracy measured by IoU."""
    image = load_fixture("screenshots/main_menu.png")
    ground_truth = load_fixture("ui/main_menu_elements.json")
    elements = detector.detect(image)
    
    matched = 0
    for gt in ground_truth:
        for pred in elements:
            if calculate_iou(gt, pred) > 0.7:
                matched += 1
                break
    
    recall = matched / len(ground_truth)
    assert recall >= 0.9  # 90% recall requirement
```

### Feature 2.4: LLM Vision Integration

```python
# tests/vision/test_llm_vision.py

@pytest.fixture
def vision_llm():
    mock_client = MockLLMClient({
        "extract game state": json.dumps({
            "resources": {"coins": 1500},
            "upgrades": [{"id": "click_power", "level": 3}],
            "current_screen": "main_gameplay"
        })
    })
    return LLMVision(client=mock_client)

def test_llm_returns_game_state(vision_llm):
    """LLM vision returns structured GameState."""
    image = load_fixture("screenshots/gameplay_early.png")
    state = vision_llm.extract_state(image)
    assert isinstance(state, GameState)
    assert "coins" in state.resources

def test_llm_caching(vision_llm):
    """Identical screenshots use cache."""
    image = load_fixture("screenshots/gameplay_early.png")
    vision_llm.extract_state(image)
    vision_llm.extract_state(image)  # Same image
    assert len(vision_llm.client.call_log) == 1  # Only one API call

def test_llm_error_retry(vision_llm):
    """Retries on API error."""
    vision_llm.client.fail_next_n(2)  # Fail first 2 calls
    image = load_fixture("screenshots/gameplay_early.png")
    state = vision_llm.extract_state(image)  # Should succeed on 3rd
    assert state is not None
    assert len(vision_llm.client.call_log) == 3

def test_set_of_mark(vision_llm):
    """Set-of-Mark annotation improves grounding."""
    image = load_fixture("screenshots/gameplay_early.png")
    state_without = vision_llm.extract_state(image, use_som=False)
    state_with = vision_llm.extract_state(image, use_som=True)
    # With SoM should have more precise element references
    assert len(state_with.ui_elements) >= len(state_without.ui_elements)
```

### Feature 3.1: Mouse Control

```python
# tests/actions/test_mouse.py

@pytest.fixture
def mouse():
    return MouseController(display=":99")

def test_move_to_position(mouse):
    """Mouse moves to specified position."""
    mouse.move_to(500, 300)
    pos = mouse.get_position()
    assert abs(pos.x - 500) < 3
    assert abs(pos.y - 300) < 3

def test_movement_is_curved(mouse):
    """Mouse movement follows curved path."""
    mouse.move_to(0, 0)
    path = mouse.move_to(1000, 1000, record_path=True)
    # Check path isn't a straight line
    midpoint_x = path[len(path)//2].x
    expected_linear = 500
    deviation = abs(midpoint_x - expected_linear)
    assert deviation > 20  # Has curvature

def test_click_timing_variance(mouse):
    """Click timing has natural variance."""
    timings = []
    for _ in range(100):
        start = time.time()
        mouse.click()
        timings.append(time.time() - start)
    
    std_dev = statistics.stdev(timings)
    assert std_dev > 0.02  # At least 20ms variance

def test_click_types(mouse):
    """Different click types work."""
    mouse.click()  # Single
    mouse.click(double=True)  # Double
    mouse.click(button="right")  # Right
    # No assertion - just verify no errors
```

### Feature 3.2: Keyboard Control

```python
# tests/actions/test_keyboard.py

@pytest.fixture
def keyboard():
    return KeyboardController(display=":99")

def test_type_text(keyboard, text_input_app):
    """Typing enters text correctly."""
    keyboard.type_text("Hello World")
    content = text_input_app.get_content()
    assert content == "Hello World"

def test_special_keys(keyboard, dialog_app):
    """Special keys work correctly."""
    keyboard.press("Escape")
    assert not dialog_app.is_open()

def test_key_combinations(keyboard, text_input_app):
    """Key combinations work."""
    keyboard.type_text("Hello")
    keyboard.combo("Ctrl", "a")  # Select all
    keyboard.type_text("Replaced")
    assert text_input_app.get_content() == "Replaced"

def test_typing_variance(keyboard):
    """Typing speed varies naturally."""
    timings = []
    for char in "The quick brown fox":
        start = time.time()
        keyboard.type_text(char)
        timings.append(time.time() - start)
    
    std_dev = statistics.stdev(timings)
    assert std_dev > 0.01  # Has variance
```

### Feature 3.3: Action Executor

```python
# tests/actions/test_executor.py

@pytest.fixture
def executor():
    return ActionExecutor(
        mouse=MockMouse(),
        keyboard=MockKeyboard(),
        vision=MockVision()
    )

def test_execute_click_action(executor):
    """Executes click action correctly."""
    action = Action(
        type=ActionType.CLICK,
        target=Point(x=500, y=300),
        parameters={}
    )
    result = executor.execute(action)
    assert result.success

def test_validate_before_click(executor):
    """Validates target exists before clicking."""
    action = Action(
        type=ActionType.CLICK,
        target=UIElement(id="nonexistent"),
        parameters={}
    )
    with pytest.raises(ActionError):
        executor.execute(action)

def test_verify_after_action(executor):
    """Verifies action had effect."""
    action = Action(
        type=ActionType.CLICK,
        target=Point(x=500, y=300),
        parameters={"expect_change": True}
    )
    result = executor.execute(action)
    assert result.state_changed

def test_rate_limiting(executor):
    """Rate limits rapid actions."""
    actions = [make_click_action() for _ in range(10)]
    start = time.time()
    for action in actions:
        executor.execute(action)
    elapsed = time.time() - start
    assert elapsed >= 1.0  # At least 100ms between actions
```

---

## Performance Test Specifications

### Vision Performance

```python
# tests/performance/test_vision_perf.py

def test_capture_latency():
    """Screenshot capture p95 < 50ms."""
    capture = ScreenCapture(display=":99")
    latencies = []
    for _ in range(1000):
        start = time.time()
        capture.capture()
        latencies.append(time.time() - start)
    
    p95 = np.percentile(latencies, 95)
    assert p95 < 0.05

def test_observation_latency():
    """Full observation p95 < 2s."""
    pipeline = ObservationPipeline(...)
    latencies = []
    for _ in range(100):
        start = time.time()
        pipeline.observe()
        latencies.append(time.time() - start)
    
    p95 = np.percentile(latencies, 95)
    assert p95 < 2.0
```

### Memory Performance

```python
# tests/performance/test_memory_perf.py

def test_memory_stability():
    """Memory stable over 1000 iterations."""
    agent = create_agent()
    
    initial_memory = get_memory_usage()
    for _ in range(1000):
        agent.step()
    final_memory = get_memory_usage()
    
    growth = final_memory - initial_memory
    assert growth < 100 * 1024 * 1024  # < 100MB growth
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: make install
      - name: Run unit tests
        run: make test-unit
      - name: Upload coverage
        uses: codecov/codecov-action@v3
  
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4
      - name: Build container
        run: docker build -t agent-zero .
      - name: Run integration tests
        run: make test-integration

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
      - name: Install dev dependencies
        run: pip install ruff mypy
      - name: Lint
        run: make lint
      - name: Type check
        run: make typecheck
```

---

## Test Coverage Requirements

| Component | Minimum Coverage |
|-----------|-----------------|
| models/ | 100% |
| interfaces/ | 100% |
| vision/ | 80% |
| actions/ | 80% |
| core/ | 80% |
| memory/ | 80% |
| strategy/ | 70% |
| observer/ | 70% |

### Coverage Enforcement

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = "--cov=src --cov-fail-under=80"
```
