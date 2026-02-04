"""Decision models for representing agent decision-making output."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, Field

from src.models.actions import Action


class Decision(BaseModel):
    """A decision made by the agent.

    Decisions capture the reasoning process and selected action.
    They are immutable records of agent decision-making.
    """

    reasoning: str = Field(
        ...,
        min_length=1,
        description="Chain-of-thought reasoning explaining the decision",
    )
    action: Action = Field(..., description="The action to take")
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ..., description="Confidence in this decision (0.0 to 1.0)"
    )
    expected_outcome: str = Field(
        ...,
        min_length=1,
        description="What we expect to happen after this action",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When decision was made",
    )
    alternatives: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative actions considered",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context used in decision",
    )

    model_config = {"frozen": True}

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence decision (>0.8)."""
        return self.confidence >= 0.8

    @property
    def is_low_confidence(self) -> bool:
        """Check if this is a low-confidence decision (<0.5)."""
        return self.confidence < 0.5
