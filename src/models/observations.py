"""Observation models for representing captured game state."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field

from src.models.game_state import GameState


class Observation(BaseModel):
    """A complete observation from the game at a point in time.

    Observations combine raw screenshot data with processed game state.
    They are immutable snapshots of what the agent saw.
    """

    screenshot: bytes = Field(..., description="Raw screenshot bytes (PNG or JPEG)")
    game_state: GameState = Field(..., description="Processed game state")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When observation was captured",
    )
    screenshot_width: Annotated[int, Field(gt=0)] = Field(
        default=1920, description="Screenshot width in pixels"
    )
    screenshot_height: Annotated[int, Field(gt=0)] = Field(
        default=1080, description="Screenshot height in pixels"
    )
    processing_time_ms: Annotated[float, Field(ge=0)] = Field(
        default=0.0, description="Time to process observation"
    )
    error: str | None = Field(default=None, description="Error message if processing failed")

    model_config = {"frozen": True}

    @property
    def is_valid(self) -> bool:
        """Check if observation was processed without errors."""
        return self.error is None

    @property
    def screenshot_size(self) -> tuple[int, int]:
        """Get screenshot dimensions as (width, height)."""
        return (self.screenshot_width, self.screenshot_height)
