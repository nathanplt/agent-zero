"""Game state models for representing the current state of the game."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field


class ScreenType(str, Enum):
    """Types of screens in the game."""

    MAIN = "main"
    MENU = "menu"
    SHOP = "shop"
    INVENTORY = "inventory"
    SETTINGS = "settings"
    LOADING = "loading"
    DIALOG = "dialog"
    PRESTIGE = "prestige"
    UNKNOWN = "unknown"


class Resource(BaseModel):
    """A game resource with name and amount."""

    name: str = Field(..., min_length=1, description="Resource name")
    amount: float = Field(..., ge=0, description="Current amount")
    max_amount: float | None = Field(default=None, ge=0, description="Maximum capacity")
    rate: float | None = Field(default=None, description="Generation rate per second")

    model_config = {"frozen": True}


class Upgrade(BaseModel):
    """A purchasable upgrade in the game."""

    id: str = Field(..., min_length=1, description="Unique upgrade identifier")
    name: str = Field(..., min_length=1, description="Display name")
    description: str | None = Field(default=None, description="Upgrade description")
    level: int = Field(default=0, ge=0, description="Current level")
    max_level: int | None = Field(default=None, ge=1, description="Maximum level")
    cost: dict[str, float] = Field(default_factory=dict, description="Cost per resource")
    effect: dict[str, Any] = Field(default_factory=dict, description="Effect description")
    available: bool = Field(default=True, description="Can be purchased now")

    model_config = {"frozen": True}

    @property
    def is_maxed(self) -> bool:
        """Check if upgrade is at max level."""
        if self.max_level is None:
            return False
        return self.level >= self.max_level


class UIElement(BaseModel):
    """A detected UI element in the game."""

    element_type: str = Field(..., description="Type of element (button, text, etc.)")
    x: Annotated[int, Field(ge=0)] = Field(..., description="Left coordinate")
    y: Annotated[int, Field(ge=0)] = Field(..., description="Top coordinate")
    width: Annotated[int, Field(gt=0)] = Field(..., description="Width in pixels")
    height: Annotated[int, Field(gt=0)] = Field(..., description="Height in pixels")
    label: str | None = Field(default=None, description="Text label if any")
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=1.0, description="Detection confidence"
    )
    clickable: bool = Field(default=True, description="Whether element is interactive")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional data")

    model_config = {"frozen": True}

    @property
    def center(self) -> tuple[int, int]:
        """Get the center point of this element."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Get bounds as (x, y, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class GameState(BaseModel):
    """Complete game state at a point in time."""

    resources: dict[str, Resource] = Field(
        default_factory=dict, description="Current resources by name"
    )
    upgrades: list[Upgrade] = Field(default_factory=list, description="Available upgrades")
    current_screen: ScreenType = Field(
        default=ScreenType.UNKNOWN, description="Current screen type"
    )
    ui_elements: list[UIElement] = Field(
        default_factory=list, description="Detected UI elements"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When state was captured"
    )
    raw_text: list[str] = Field(default_factory=list, description="Raw OCR text extracted")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional state data")

    model_config = {"frozen": True}

    def get_resource(self, name: str) -> Resource | None:
        """Get a resource by name."""
        return self.resources.get(name)

    def get_upgrade(self, upgrade_id: str) -> Upgrade | None:
        """Get an upgrade by ID."""
        for upgrade in self.upgrades:
            if upgrade.id == upgrade_id:
                return upgrade
        return None

    def find_element(self, element_type: str, label: str | None = None) -> UIElement | None:
        """Find a UI element by type and optional label."""
        for element in self.ui_elements:
            if element.element_type == element_type:
                if label is None or element.label == label:
                    return element
        return None

    def find_elements(self, element_type: str) -> list[UIElement]:
        """Find all UI elements of a given type."""
        return [e for e in self.ui_elements if e.element_type == element_type]
