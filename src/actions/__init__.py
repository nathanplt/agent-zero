"""Actions package for controlling game input.

This package provides:
- MouseController: Human-like mouse movement and clicks
- KeyboardController: Human-like keyboard input with natural timing
- Action executor (future)
"""

from src.actions.keyboard import KeyboardController
from src.actions.mouse import MouseController

__all__ = ["KeyboardController", "MouseController"]
