"""Action models for representing agent actions."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, Field


class ActionType(StrEnum):
    """Types of actions the agent can perform."""

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    TYPE = "type"
    KEY_PRESS = "key_press"
    KEY_COMBO = "key_combo"
    SCROLL = "scroll"
    WAIT = "wait"
    MOVE = "move"


class Point(BaseModel):
    """A point in screen coordinates."""

    x: Annotated[int, Field(ge=0)] = Field(..., description="X coordinate")
    y: Annotated[int, Field(ge=0)] = Field(..., description="Y coordinate")

    model_config = {"frozen": True}


class Action(BaseModel):
    """An action to be performed by the agent.

    Actions are immutable and represent a single atomic operation.
    """

    type: ActionType = Field(..., description="The type of action to perform")
    target: Point | None = Field(default=None, description="Target coordinates for click/move")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional action parameters",
    )
    description: str | None = Field(default=None, description="Human-readable description")

    model_config = {"frozen": True}

    @classmethod
    def click(
        cls,
        x: int,
        y: int,
        button: str = "left",
        description: str | None = None,
    ) -> Action:
        """Create a click action."""
        return cls(
            type=ActionType.CLICK,
            target=Point(x=x, y=y),
            parameters={"button": button},
            description=description,
        )

    @classmethod
    def double_click(cls, x: int, y: int, description: str | None = None) -> Action:
        """Create a double-click action."""
        return cls(
            type=ActionType.DOUBLE_CLICK,
            target=Point(x=x, y=y),
            description=description,
        )

    @classmethod
    def right_click(cls, x: int, y: int, description: str | None = None) -> Action:
        """Create a right-click action."""
        return cls(
            type=ActionType.RIGHT_CLICK,
            target=Point(x=x, y=y),
            description=description,
        )

    @classmethod
    def type_text(cls, text: str, description: str | None = None) -> Action:
        """Create a type text action."""
        return cls(
            type=ActionType.TYPE,
            parameters={"text": text},
            description=description,
        )

    @classmethod
    def key_press(cls, key: str, description: str | None = None) -> Action:
        """Create a key press action."""
        return cls(
            type=ActionType.KEY_PRESS,
            parameters={"key": key},
            description=description,
        )

    @classmethod
    def key_combo(cls, keys: list[str], description: str | None = None) -> Action:
        """Create a key combination action."""
        return cls(
            type=ActionType.KEY_COMBO,
            parameters={"keys": keys},
            description=description,
        )

    @classmethod
    def scroll(
        cls,
        direction: str,
        amount: int = 1,
        description: str | None = None,
    ) -> Action:
        """Create a scroll action."""
        return cls(
            type=ActionType.SCROLL,
            parameters={"direction": direction, "amount": amount},
            description=description,
        )

    @classmethod
    def wait(cls, duration_ms: int, description: str | None = None) -> Action:
        """Create a wait action."""
        return cls(
            type=ActionType.WAIT,
            parameters={"duration_ms": duration_ms},
            description=description,
        )

    @classmethod
    def move(cls, x: int, y: int, description: str | None = None) -> Action:
        """Create a mouse move action."""
        return cls(
            type=ActionType.MOVE,
            target=Point(x=x, y=y),
            description=description,
        )
