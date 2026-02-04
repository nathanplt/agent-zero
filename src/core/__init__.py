"""Core agent logic package.

This package provides:
- ObservationPipeline: Orchestrates vision components to produce observations
- Observation: Complete observation with screenshot, game state, and UI elements
- DecisionEngine: ReAct-style reasoning and action selection
- DecisionConfig: Configuration for the decision engine
- DecisionPrompts: Prompt templates for decision making
- AgentLoop: Main observe-decide-act loop
- LoopConfig: Configuration for the agent loop
- LoopState: Loop state enumeration
- AgentMetrics: Snapshot of collected metrics
- MetricsCollector: Metrics collection for monitoring
- RecoverableError: Error that allows loop to continue
- FatalError: Error that requires stopping the loop
"""

from src.core.decision import DecisionConfig, DecisionEngine, DecisionEngineError
from src.core.loop import (
    AgentLoop,
    FatalError,
    LoopConfig,
    LoopState,
    RecoverableError,
)
from src.core.metrics import AgentMetrics, MetricsCollector
from src.core.observation import Observation, ObservationPipeline
from src.core.prompts import DecisionPrompts

__all__ = [
    "AgentLoop",
    "AgentMetrics",
    "DecisionConfig",
    "DecisionEngine",
    "DecisionEngineError",
    "DecisionPrompts",
    "FatalError",
    "LoopConfig",
    "LoopState",
    "MetricsCollector",
    "Observation",
    "ObservationPipeline",
    "RecoverableError",
]
