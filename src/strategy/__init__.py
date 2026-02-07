"""Strategy package: goal hierarchy and game-specific intelligence."""

from src.strategy.adapters.target_game import TargetGameAdapter
from src.strategy.goals import GoalManager
from src.strategy.incremental import IncrementalMetaStrategy
from src.strategy.planning import PlanningSystem
from src.strategy.policy import (
    IncrementalPolicyConfig,
    IncrementalPolicyEngine,
    PolicyProposal,
    parse_compact_number,
)

__all__ = [
    "GoalManager",
    "IncrementalMetaStrategy",
    "PlanningSystem",
    "TargetGameAdapter",
    "IncrementalPolicyConfig",
    "IncrementalPolicyEngine",
    "PolicyProposal",
    "parse_compact_number",
]
