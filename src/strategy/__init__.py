"""Strategy package: goal hierarchy and game-specific intelligence."""

from src.strategy.goals import GoalManager
from src.strategy.incremental import IncrementalMetaStrategy

__all__ = ["GoalManager", "IncrementalMetaStrategy"]
