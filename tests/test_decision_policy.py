"""Policy integration tests for DecisionEngine."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock

from src.core.decision import DecisionConfig, DecisionEngine
from src.core.observation import Observation
from src.interfaces.actions import ActionType
from src.interfaces.vision import Screenshot
from src.models.actions import Action, Point
from src.models.game_state import GameState, Resource, ScreenType
from src.strategy.policy import PolicyProposal


def _observation() -> Observation:
    screenshot = MagicMock(spec=Screenshot)
    screenshot.width = 1920
    screenshot.height = 1080
    screenshot.timestamp = datetime.now()

    return Observation(
        screenshot=screenshot,
        game_state=GameState(
            current_screen=ScreenType.MAIN,
            resources={"money": Resource(name="money", amount=5000.0)},
        ),
        ui_elements=[],
        text_regions=[],
        timestamp=datetime.now(),
        confidence=0.9,
    )


class TestDecisionEnginePolicyFlow:
    """DecisionEngine should use deterministic policy before LLM."""

    def test_uses_policy_when_high_confidence(self) -> None:
        engine = DecisionEngine(
            config=DecisionConfig(
                policy_enabled=True,
                policy_confidence_threshold=0.7,
            )
        )
        policy = MagicMock()
        policy.propose.return_value = PolicyProposal(
            action=Action(
                type=ActionType.CLICK,
                target=Point(x=100, y=200),
                description="Click upgrade",
            ),
            confidence=0.92,
            strategy="upgrade",
            rationale="Affordable upgrade available.",
            expected_outcome="Resource generation should increase.",
        )
        engine._policy = policy
        engine._call_llm = MagicMock(side_effect=AssertionError("LLM should not be called"))

        decision = engine.decide(_observation(), use_cache=False)

        assert decision.action.type == ActionType.CLICK
        assert decision.context.get("decision_source") == "policy"
        assert decision.confidence == 0.92
        assert "Affordable upgrade" in decision.reasoning

    def test_falls_back_to_llm_when_policy_confidence_low(self) -> None:
        engine = DecisionEngine(
            config=DecisionConfig(
                policy_enabled=True,
                policy_confidence_threshold=0.7,
            )
        )
        policy = MagicMock()
        policy.propose.return_value = PolicyProposal(
            action=Action(
                type=ActionType.WAIT,
                parameters={"duration_ms": 800},
                description="Low-confidence fallback",
            ),
            confidence=0.4,
            strategy="safe_wait",
            rationale="Not enough certainty.",
            expected_outcome="Will observe again.",
        )
        engine._policy = policy

        engine._call_llm = MagicMock(
            return_value=json.dumps(
                {
                    "thought": "Need to click for progress",
                    "action": {
                        "type": "click",
                        "target": {"x": 250, "y": 350},
                        "parameters": {},
                        "description": "Click center",
                    },
                    "confidence": 0.8,
                    "expected_outcome": "Progress increases",
                    "alternatives": [],
                }
            )
        )

        decision = engine.decide(_observation(), use_cache=False)

        assert decision.action.type == ActionType.CLICK
        assert decision.context.get("decision_source") == "llm"

    def test_policy_disabled_uses_llm_only(self) -> None:
        engine = DecisionEngine(config=DecisionConfig(policy_enabled=False))
        policy = MagicMock()
        engine._policy = policy
        engine._call_llm = MagicMock(
            return_value=json.dumps(
                {
                    "thought": "Wait for resource tick",
                    "action": {
                        "type": "wait",
                        "target": None,
                        "parameters": {"duration_ms": 1000},
                        "description": "Wait",
                    },
                    "confidence": 0.7,
                    "expected_outcome": "More resources",
                    "alternatives": [],
                }
            )
        )

        decision = engine.decide(_observation(), use_cache=False)

        assert decision.action.type == ActionType.WAIT
        assert decision.context.get("decision_source") == "llm"
        policy.propose.assert_not_called()
