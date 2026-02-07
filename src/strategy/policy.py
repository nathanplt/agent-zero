"""Generalized policy engine for incremental/clicker game action selection.

This module provides a deterministic, low-latency policy layer that can pick
high-value actions from structured observations before falling back to an LLM.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.interfaces.actions import ActionType
from src.models.actions import Action, Point

if TYPE_CHECKING:
    from src.core.observation import Observation
    from src.interfaces.vision import UIElement

_NUMBER_PATTERN = re.compile(
    r"(?P<num>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?P<suffix>[kmbtq]?)",
    re.IGNORECASE,
)
_MULTIPLIERS: dict[str, float] = {
    "": 1.0,
    "k": 1_000.0,
    "m": 1_000_000.0,
    "b": 1_000_000_000.0,
    "t": 1_000_000_000_000.0,
    "q": 1_000_000_000_000_000.0,
}

_DIALOG_CONFIRM_KEYWORDS = frozenset(
    {
        "ok",
        "yes",
        "confirm",
        "continue",
        "accept",
        "claim",
        "collect",
    }
)
_DIALOG_CANCEL_KEYWORDS = frozenset(
    {
        "no",
        "cancel",
        "close",
        "dismiss",
        "later",
    }
)
_PRESTIGE_KEYWORDS = frozenset({"prestige", "rebirth", "ascend", "reset", "transcend"})
_UPGRADE_KEYWORDS = frozenset({"buy", "upgrade", "level", "boost", "x", "multiplier", "tier"})
_MAIN_CLICK_KEYWORDS = frozenset({"click", "tap", "main", "collect", "earn"})


def parse_compact_number(text: str) -> float | None:
    """Parse compact number text (e.g. 1.5K, 2M, 1,250)."""
    cleaned = text.strip().replace(",", "")
    if not cleaned:
        return None

    match = _NUMBER_PATTERN.fullmatch(cleaned)
    if not match:
        return None

    number = float(match.group("num"))
    suffix = match.group("suffix").lower()
    multiplier = _MULTIPLIERS.get(suffix)
    if multiplier is None:
        return None
    return number * multiplier


@dataclass(frozen=True)
class IncrementalPolicyConfig:
    """Configuration for policy behavior."""

    default_prestige_threshold: float = 1_000_000_000.0
    max_repeated_failures: int = 3
    min_upgrade_confidence: float = 0.65
    min_main_click_confidence: float = 0.6


@dataclass(frozen=True)
class PolicyProposal:
    """Policy-selected action and supporting metadata."""

    action: Action
    confidence: float
    strategy: str
    rationale: str
    expected_outcome: str


@dataclass(frozen=True)
class _Candidate:
    """Internal scoring candidate."""

    action: Action
    strategy: str
    rationale: str
    expected_outcome: str
    score: float
    confidence: float


class IncrementalPolicyEngine:
    """Rule-based policy for incremental games with lightweight scoring."""

    def __init__(self, config: IncrementalPolicyConfig | None = None) -> None:
        self._config = config or IncrementalPolicyConfig()

    def propose(
        self,
        observation: Observation,
        recent_actions: list[dict[str, Any]] | None = None,
    ) -> PolicyProposal:
        """Propose an action from the current observation.

        The policy prioritizes:
        1) dialog confirmations
        2) prestige when threshold conditions are met
        3) affordable upgrades with highest utility score
        4) main click area interaction
        5) safe wait fallback
        """
        recent_actions = recent_actions or []
        ui_elements = list(observation.ui_elements)
        resources = {
            name: resource.amount for name, resource in observation.game_state.resources.items()
        }
        metadata = observation.game_state.metadata

        candidates: list[_Candidate] = []

        dialog_candidate = self._dialog_candidate(ui_elements, observation.game_state.current_screen.value)
        if dialog_candidate is not None:
            candidates.append(dialog_candidate)

        prestige_candidate = self._prestige_candidate(ui_elements, resources, metadata)
        if prestige_candidate is not None:
            candidates.append(prestige_candidate)

        upgrade_candidates = self._upgrade_candidates(ui_elements, resources, metadata)
        candidates.extend(upgrade_candidates)

        main_click_candidate = self._main_click_candidate(ui_elements)
        if main_click_candidate is not None:
            candidates.append(main_click_candidate)

        wait_candidate = _Candidate(
            action=Action(
                type=ActionType.WAIT,
                parameters={"duration_ms": 800, "strategy": "safe_wait"},
                description="Wait for more state changes",
            ),
            strategy="safe_wait",
            rationale="No reliable high-value interaction detected, waiting briefly.",
            expected_outcome="New UI/resource state should appear.",
            score=0.1,
            confidence=0.35,
        )
        candidates.append(wait_candidate)

        penalties = self._failed_strategy_penalties(recent_actions)
        best = self._pick_best_candidate(candidates, penalties)

        return PolicyProposal(
            action=best.action,
            confidence=max(0.0, min(1.0, best.confidence)),
            strategy=best.strategy,
            rationale=best.rationale,
            expected_outcome=best.expected_outcome,
        )

    def _pick_best_candidate(
        self,
        candidates: list[_Candidate],
        penalties: dict[str, float],
    ) -> _Candidate:
        def effective_score(candidate: _Candidate) -> float:
            return candidate.score - penalties.get(candidate.strategy, 0.0)

        return max(candidates, key=effective_score)

    def _failed_strategy_penalties(self, recent_actions: list[dict[str, Any]]) -> dict[str, float]:
        failures: dict[str, int] = {}
        for action in recent_actions:
            if str(action.get("result", "")).lower() != "failed":
                continue
            strategy = str(action.get("strategy", "")).strip()
            if not strategy:
                continue
            failures[strategy] = failures.get(strategy, 0) + 1

        penalties: dict[str, float] = {}
        for strategy, count in failures.items():
            if count >= self._config.max_repeated_failures:
                penalties[strategy] = 10.0
        return penalties

    def _dialog_candidate(self, ui_elements: list[UIElement], screen_name: str) -> _Candidate | None:
        is_dialog_like = screen_name == "dialog"
        positive: list[UIElement] = []
        negative: list[UIElement] = []

        for element in ui_elements:
            if not element.clickable:
                continue
            label = (element.label or "").strip().lower()
            if not label:
                continue
            if any(keyword in label for keyword in _DIALOG_CONFIRM_KEYWORDS):
                positive.append(element)
                is_dialog_like = True
            elif any(keyword in label for keyword in _DIALOG_CANCEL_KEYWORDS):
                negative.append(element)
                is_dialog_like = True

        if not is_dialog_like:
            return None

        target = positive[0] if positive else (negative[0] if negative else None)
        if target is None:
            return None

        center_x = target.x + target.width // 2
        center_y = target.y + target.height // 2
        description = (
            f"Confirm dialog action via '{target.label}'"
            if target in positive
            else f"Dismiss dialog via '{target.label}'"
        )
        return _Candidate(
            action=Action(
                type=ActionType.CLICK,
                target=Point(x=center_x, y=center_y),
                parameters={
                    "strategy": "dialog_confirm" if target in positive else "dialog_dismiss"
                },
                description=description,
            ),
            strategy="dialog_confirm" if target in positive else "dialog_dismiss",
            rationale="Dialog blocks progress; resolving it is the highest-priority action.",
            expected_outcome="Dialog closes and gameplay resumes.",
            score=5.0,
            confidence=0.95,
        )

    def _prestige_candidate(
        self,
        ui_elements: list[UIElement],
        resources: dict[str, float],
        metadata: dict[str, Any],
    ) -> _Candidate | None:
        prestige_button: UIElement | None = None
        for element in ui_elements:
            if not element.clickable:
                continue
            label = (element.label or "").lower()
            if any(keyword in label for keyword in _PRESTIGE_KEYWORDS):
                prestige_button = element
                break

        if prestige_button is None:
            return None

        primary_resource_name = str(metadata.get("primary_resource") or self._primary_resource_name(resources))
        primary_value = resources.get(primary_resource_name, 0.0)
        if primary_value <= 0 and resources:
            primary_value = max(resources.values())

        threshold_value = float(
            metadata.get("prestige_threshold", self._config.default_prestige_threshold)
        )
        if primary_value < threshold_value:
            return None

        center_x = prestige_button.x + prestige_button.width // 2
        center_y = prestige_button.y + prestige_button.height // 2
        return _Candidate(
            action=Action(
                type=ActionType.CLICK,
                target=Point(x=center_x, y=center_y),
                parameters={"strategy": "prestige"},
                description=f"Trigger prestige/rebirth via '{prestige_button.label}'",
            ),
            strategy="prestige",
            rationale=(
                f"Primary resource {primary_resource_name} reached prestige threshold "
                f"({primary_value:.0f} >= {threshold_value:.0f})."
            ),
            expected_outcome="Prestige multiplier should improve long-run progression.",
            score=6.0,
            confidence=0.9,
        )

    def _upgrade_candidates(
        self,
        ui_elements: list[UIElement],
        resources: dict[str, float],
        metadata: dict[str, Any],
    ) -> list[_Candidate]:
        if not resources:
            return []

        primary_resource_name = str(metadata.get("primary_resource") or self._primary_resource_name(resources))
        available = resources.get(primary_resource_name, 0.0)
        if available <= 0:
            available = max(resources.values()) if resources else 0.0

        candidates: list[_Candidate] = []
        for element in ui_elements:
            if not element.clickable:
                continue
            label = (element.label or "").strip()
            if not label:
                continue
            lowered = label.lower()
            if any(keyword in lowered for keyword in _PRESTIGE_KEYWORDS):
                continue
            if not self._looks_like_upgrade(lowered):
                continue

            cost = self._extract_cost(label)
            if cost is None or cost <= 0:
                # Unknown cost upgrades are treated as potentially affordable.
                cost = available * 0.9 if available > 0 else 1.0

            if cost > available:
                continue

            utility = self._estimate_upgrade_utility(label, cost, available)
            center_x = element.x + element.width // 2
            center_y = element.y + element.height // 2
            score = 2.5 + utility
            confidence = max(self._config.min_upgrade_confidence, min(0.92, 0.65 + utility / 5.0))
            candidates.append(
                _Candidate(
                    action=Action(
                        type=ActionType.CLICK,
                        target=Point(x=center_x, y=center_y),
                        parameters={"strategy": "upgrade"},
                        description=f"Purchase upgrade '{label}'",
                    ),
                    strategy="upgrade",
                    rationale=(
                        f"Affordable upgrade detected ({primary_resource_name}: "
                        f"{available:.0f}, cost: {cost:.0f}); selecting highest utility option."
                    ),
                    expected_outcome="Upgrade should increase resource generation.",
                    score=score,
                    confidence=confidence,
                )
            )

        return candidates

    def _main_click_candidate(self, ui_elements: list[UIElement]) -> _Candidate | None:
        preferred: UIElement | None = None
        fallback: UIElement | None = None
        largest_area = -1

        for element in ui_elements:
            if not element.clickable:
                continue
            label = (element.label or "").lower()
            area = element.width * element.height

            if any(keyword in label for keyword in _MAIN_CLICK_KEYWORDS):
                preferred = element
                break

            # Generic fallback: largest central clickable region.
            center_x = element.x + element.width // 2
            center_y = element.y + element.height // 2
            if self._is_centerish(center_x, center_y) and area > largest_area:
                largest_area = area
                fallback = element

        target = preferred or fallback
        if target is None:
            return None

        center_x = target.x + target.width // 2
        center_y = target.y + target.height // 2
        return _Candidate(
            action=Action(
                type=ActionType.CLICK,
                target=Point(x=center_x, y=center_y),
                parameters={"strategy": "main_click"},
                description="Click the main progression area",
            ),
            strategy="main_click",
            rationale="No better immediate purchase available; continue core resource loop.",
            expected_outcome="Primary resource should increase.",
            score=1.2,
            confidence=self._config.min_main_click_confidence,
        )

    @staticmethod
    def _extract_cost(label: str) -> float | None:
        tokens = label.replace(",", " ")
        matches = _NUMBER_PATTERN.findall(tokens)
        if not matches:
            return None
        parsed = [parse_compact_number(f"{number}{suffix}") for number, suffix in matches]
        parsed_values: list[float] = [value for value in parsed if value is not None]
        if not parsed_values:
            return None
        # Cost is usually the largest value in an upgrade label.
        return max(parsed_values)

    @staticmethod
    def _estimate_upgrade_utility(label: str, cost: float, available: float) -> float:
        multiplier = 1.0
        label_lower = label.lower()
        mult_match = re.search(r"x\s*(\d+(?:\.\d+)?)", label_lower)
        if mult_match:
            multiplier = max(1.0, float(mult_match.group(1)))

        # Prefer cheaper upgrades when utility is similar to keep progression smooth.
        affordability_ratio = available / max(cost, 1.0)
        return math.log1p(multiplier) + min(2.0, affordability_ratio / 2.0)

    @staticmethod
    def _looks_like_upgrade(label_lower: str) -> bool:
        if any(keyword in label_lower for keyword in _UPGRADE_KEYWORDS):
            return True
        return _NUMBER_PATTERN.search(label_lower.replace(",", "")) is not None

    @staticmethod
    def _primary_resource_name(resources: dict[str, float]) -> str:
        if not resources:
            return "money"
        return max(resources, key=resources.__getitem__)

    @staticmethod
    def _is_centerish(x: int, y: int, width: int = 1920, height: int = 1080) -> bool:
        return width * 0.2 <= x <= width * 0.8 and height * 0.2 <= y <= height * 0.8
