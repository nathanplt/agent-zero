"""Incremental game meta-strategy (Feature 6.2).

Prestige timing, resource allocation, and upgrade prioritization
for incremental/clicker games. Adapts to game state and optional game config.
"""

from __future__ import annotations

from typing import Any

from src.interfaces.strategy import ResourceAllocation, UpgradePriority


class IncrementalMetaStrategy:
    """Knowledge about optimal incremental game progression.

    - recommend_resource_allocation: how to spend current resources
    - recommend_prestige_timing: when to prestige/rebirth
    - prioritize_upgrades: order upgrades by ROI (benefit/cost)
    """

    def __init__(self) -> None:
        """Initialize with default rules; can be extended with game config later."""
        pass

    def recommend_resource_allocation(
        self,
        resources: dict[str, float],
        game_state: dict[str, Any],
    ) -> ResourceAllocation:
        """Recommend how to allocate available resources.

        Favors primary resource (from game_state or first key). Allocates
        a large fraction to upgrades and reserves some for prestige.
        """
        primary = game_state.get("primary") or (list(resources.keys())[0] if resources else "money")
        allocations: dict[str, float] = {}
        for name, amount in resources.items():
            if name == primary:
                allocations[name] = amount * 0.8
            else:
                allocations[name] = amount * 0.2
        reasoning = (
            f"Allocate ~80% of {primary} toward upgrades for growth, "
            "~20% reserved for prestige or safety."
        )
        expected_roi = 1.2
        return ResourceAllocation(
            allocations=allocations,
            reasoning=reasoning,
            expected_roi=expected_roi,
        )

    def recommend_prestige_timing(self, game_state: dict[str, Any]) -> tuple[bool, str]:
        """Recommend whether to prestige now.

        Recommends prestige when prestige_level is near prestige_target,
        or when progress has slowed (optional growth rate in game_state).
        """
        prestige_level = game_state.get("prestige_level", 0)
        prestige_target = game_state.get("prestige_target", 10)
        resources = game_state.get("resources") or {}
        primary_total = sum(resources.values()) or 0

        if prestige_level >= prestige_target:
            return False, "Already at or past prestige target; no need to prestige again."

        if primary_total >= 1e9 and prestige_level < prestige_target - 1:
            return True, "High resource accumulation; prestige to multiply long-term progress."
        if prestige_level >= prestige_target - 2 and primary_total >= 1e6:
            return True, "Near prestige target; prestige to unlock next tier."

        return False, "Continue accumulating; prestige later for better multiplier."

    def prioritize_upgrades(
        self,
        available_upgrades: list[dict[str, Any]],
        resources: dict[str, float],
    ) -> list[UpgradePriority]:
        """Prioritize available upgrades by ROI (benefit/cost); affordable first when tied."""
        total_resources = sum(resources.values()) or 1.0
        scored: list[tuple[float, dict[str, Any], float]] = []

        for u in available_upgrades:
            cost = u.get("cost") or {}
            if isinstance(cost, dict):
                total_cost = sum(cost.values()) or 1.0
            else:
                total_cost = float(cost) if cost else 1.0
            benefit = float(u.get("benefit", 0))
            roi = benefit / total_cost if total_cost > 0 else benefit
            affordable = 1.0 if total_cost <= total_resources else 0.0
            priority_score = roi * (1 + 0.5 * affordable)
            scored.append((priority_score, u, benefit))

        scored.sort(key=lambda x: (-x[0], x[1].get("id", "")))
        result: list[UpgradePriority] = []
        for priority_score, u, benefit in scored:
            cost = u.get("cost") or {}
            if not isinstance(cost, dict):
                cost = {"value": float(cost)}
            result.append(
                UpgradePriority(
                    upgrade_id=str(u.get("id", "")),
                    name=str(u.get("name", "")),
                    cost={k: float(v) for k, v in cost.items()},
                    expected_benefit=benefit,
                    priority_score=priority_score,
                )
            )
        return result
