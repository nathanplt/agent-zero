"""Actions package for controlling game input.

This package provides:
- MouseController: Human-like mouse movement and clicks
- KeyboardController: Human-like keyboard input with natural timing
- InputBackend: Interface for actual input mechanisms
- PlaywrightInputBackend: Backend using Playwright's page.keyboard/mouse
- NullInputBackend: No-op backend for testing
"""

from src.actions.backend import InputBackend, NullInputBackend, PlaywrightInputBackend
from src.actions.keyboard import KeyboardController
from src.actions.mouse import MouseController

__all__ = [
    "InputBackend",
    "KeyboardController",
    "MouseController",
    "NullInputBackend",
    "PlaywrightInputBackend",
]
