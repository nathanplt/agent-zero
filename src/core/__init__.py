"""Core agent logic package.

This package provides:
- ObservationPipeline: Orchestrates vision components to produce observations
- Observation: Complete observation with screenshot, game state, and UI elements
"""

from src.core.observation import Observation, ObservationPipeline

__all__ = [
    "Observation",
    "ObservationPipeline",
]
