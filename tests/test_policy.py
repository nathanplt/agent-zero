"""Tests for generalized incremental-game policy engine."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from src.core.observation import Observation
from src.interfaces.actions import ActionType
from src.interfaces.vision import Screenshot
from src.interfaces.vision import UIElement as VisionUIElement
from src.models.game_state import GameState, Resource, ScreenType
from src.models.game_state import UIElement as ModelUIElement
from src.strategy.policy import (
    IncrementalPolicyConfig,
    IncrementalPolicyEngine,
    parse_compact_number,
)


def _element(
    label: str,
    *,
    x: int,
    y: int,
    width: int = 120,
    height: int = 40,
    element_type: str = "button",
    clickable: bool = True,
) -> VisionUIElement:
    return VisionUIElement(
        element_type=element_type,
        x=x,
        y=y,
        width=width,
        height=height,
        confidence=0.95,
        label=label,
        clickable=clickable,
    )


def _observation(
    *,
    resources: dict[str, float] | None = None,
    ui_elements: list[VisionUIElement] | None = None,
    screen: ScreenType = ScreenType.MAIN,
    metadata: dict[str, object] | None = None,
) -> Observation:
    screenshot = MagicMock(spec=Screenshot)
    screenshot.width = 1920
    screenshot.height = 1080
    screenshot.timestamp = datetime.now()

    resources = resources or {}
    ui_elements = ui_elements or []

    model_elements = [
        ModelUIElement(
            element_type=e.element_type,
            x=e.x,
            y=e.y,
            width=e.width,
            height=e.height,
            label=e.label,
            confidence=e.confidence,
            clickable=e.clickable,
            metadata=e.metadata,
        )
        for e in ui_elements
    ]

    game_state = GameState(
        current_screen=screen,
        resources={k: Resource(name=k, amount=v) for k, v in resources.items()},
        ui_elements=model_elements,
        metadata=metadata or {},
    )

    return Observation(
        screenshot=screenshot,
        game_state=game_state,
        ui_elements=ui_elements,
        text_regions=[],
        timestamp=datetime.now(),
    )


class TestCompactNumberParsing:
    """Tests for compact number parsing used by policy/adapter scoring."""

    def test_parses_plain_numbers_and_commas(self) -> None:
        assert parse_compact_number("1250") == 1250.0
        assert parse_compact_number("1,250") == 1250.0

    def test_parses_suffix_notation(self) -> None:
        assert parse_compact_number("1.5K") == 1500.0
        assert parse_compact_number("2M") == 2_000_000.0
        assert parse_compact_number("3.25b") == 3_250_000_000.0

    def test_returns_none_for_invalid_text(self) -> None:
        assert parse_compact_number("not-a-number") is None


class TestPolicyDecisionSelection:
    """Tests for policy action selection and general gameplay behavior."""

    def test_returns_wait_when_no_clickable_elements(self) -> None:
        engine = IncrementalPolicyEngine()
        obs = _observation(resources={"money": 100.0}, ui_elements=[])

        proposal = engine.propose(obs)

        assert proposal.action.type == ActionType.WAIT
        assert proposal.confidence <= 0.5
        assert proposal.strategy == "safe_wait"

    def test_prioritizes_dialog_confirmation_actions(self) -> None:
        engine = IncrementalPolicyEngine()
        obs = _observation(
            resources={"money": 100.0},
            screen=ScreenType.DIALOG,
            ui_elements=[
                _element("No", x=780, y=560),
                _element("Confirm", x=920, y=560),
            ],
        )

        proposal = engine.propose(obs)

        assert proposal.action.type == ActionType.CLICK
        assert proposal.strategy == "dialog_confirm"
        assert proposal.action.target is not None
        assert proposal.action.target.x == 980
        assert proposal.action.target.y == 580

    def test_uses_prestige_when_threshold_and_button_available(self) -> None:
        engine = IncrementalPolicyEngine(
            IncrementalPolicyConfig(default_prestige_threshold=1_000_000.0)
        )
        obs = _observation(
            resources={"money": 2_500_000.0},
            ui_elements=[
                _element("Prestige", x=1500, y=120),
                _element("Buy x2 - 1.0M", x=1400, y=420),
            ],
            metadata={"primary_resource": "money"},
        )

        proposal = engine.propose(obs)

        assert proposal.action.type == ActionType.CLICK
        assert proposal.strategy == "prestige"
        assert proposal.confidence >= 0.8

    def test_avoids_prestige_when_below_threshold(self) -> None:
        engine = IncrementalPolicyEngine(
            IncrementalPolicyConfig(default_prestige_threshold=1_000_000.0)
        )
        obs = _observation(
            resources={"money": 50_000.0},
            ui_elements=[
                _element("Prestige", x=1500, y=120),
                _element("Buy x2 - 20K", x=1400, y=420),
            ],
            metadata={"primary_resource": "money"},
        )

        proposal = engine.propose(obs)

        assert proposal.strategy != "prestige"

    def test_prioritizes_affordable_upgrade_before_main_click(self) -> None:
        engine = IncrementalPolicyEngine()
        obs = _observation(
            resources={"money": 7500.0},
            ui_elements=[
                _element("Buy x1.5 - 2.0K", x=1350, y=380),
                _element("Main", x=860, y=460, width=260, height=220, element_type="panel"),
            ],
            metadata={"primary_resource": "money"},
        )

        proposal = engine.propose(obs)

        assert proposal.action.type == ActionType.CLICK
        assert proposal.strategy == "upgrade"

    def test_falls_back_to_main_click_when_upgrades_unaffordable(self) -> None:
        engine = IncrementalPolicyEngine()
        obs = _observation(
            resources={"money": 100.0},
            ui_elements=[
                _element("Buy x10 - 20K", x=1350, y=380),
                _element("Tap", x=860, y=460, width=260, height=220, element_type="panel"),
            ],
            metadata={"primary_resource": "money"},
        )

        proposal = engine.propose(obs)

        assert proposal.action.type == ActionType.CLICK
        assert proposal.strategy == "main_click"

    def test_stagnation_penalizes_repeated_failed_strategy(self) -> None:
        engine = IncrementalPolicyEngine(
            IncrementalPolicyConfig(max_repeated_failures=2)
        )
        obs = _observation(
            resources={"money": 5000.0},
            ui_elements=[
                _element("Buy x2 - 2K", x=1350, y=380),
                _element("Tap", x=860, y=460, width=260, height=220, element_type="panel"),
            ],
            metadata={"primary_resource": "money"},
        )

        recent_actions = [
            {"strategy": "upgrade", "result": "failed"},
            {"strategy": "upgrade", "result": "failed"},
        ]

        proposal = engine.propose(obs, recent_actions=recent_actions)

        assert proposal.strategy == "main_click"

    def test_uses_metadata_override_for_prestige_threshold(self) -> None:
        engine = IncrementalPolicyEngine(
            IncrementalPolicyConfig(default_prestige_threshold=1_000_000.0)
        )
        obs = _observation(
            resources={"money": 250_000.0},
            ui_elements=[_element("Rebirth", x=1500, y=120)],
            metadata={
                "primary_resource": "money",
                "prestige_threshold": 200_000.0,
            },
        )

        proposal = engine.propose(obs)

        assert proposal.strategy == "prestige"

    def test_includes_human_readable_rationale(self) -> None:
        engine = IncrementalPolicyEngine()
        obs = _observation(
            resources={"money": 1000.0},
            ui_elements=[_element("Buy x2 - 500", x=1350, y=380)],
            metadata={"primary_resource": "money"},
        )

        proposal = engine.propose(obs)

        assert isinstance(proposal.rationale, str)
        assert len(proposal.rationale) > 10
