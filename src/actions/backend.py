"""Input backend interface and implementations.

This module provides:
- InputBackend: Abstract interface for actual input mechanisms
- PlaywrightInputBackend: Implementation using Playwright's page.keyboard/mouse
- NullInputBackend: No-op implementation for testing

The backend pattern allows controllers (MouseController, KeyboardController)
to have their human-like timing logic while delegating actual input to
swappable backends.

Example:
    >>> from playwright.sync_api import sync_playwright
    >>> from src.actions.backend import PlaywrightInputBackend
    >>> from src.actions.keyboard import KeyboardController
    >>>
    >>> with sync_playwright() as p:
    ...     browser = p.chromium.launch()
    ...     page = browser.new_page()
    ...     backend = PlaywrightInputBackend(page)
    ...     keyboard = KeyboardController(backend=backend)
    ...     keyboard.type_text("Hello World")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Page

logger = logging.getLogger(__name__)


class InputBackend(ABC):
    """Abstract interface for input mechanisms.

    This interface defines the low-level input operations that controllers
    use to actually send input to the system. Different implementations
    allow for different backends (Playwright, pyautogui, etc.) or testing
    (NullInputBackend).

    Controllers contain the human-like timing logic (delays, curves);
    backends perform the actual input operations.
    """

    # Keyboard operations

    @abstractmethod
    def key_down(self, key: str) -> None:
        """Press a key down (without releasing).

        Args:
            key: Key name (e.g., 'a', 'Enter', 'Control').
        """
        ...

    @abstractmethod
    def key_up(self, key: str) -> None:
        """Release a key.

        Args:
            key: Key name.
        """
        ...

    @abstractmethod
    def type_char(self, char: str) -> None:
        """Type a single character.

        This is for printable characters. For special keys, use key_down/key_up.

        Args:
            char: Single character to type.
        """
        ...

    # Mouse operations

    @abstractmethod
    def mouse_move(self, x: int, y: int) -> None:
        """Move the mouse to absolute coordinates.

        Args:
            x: X coordinate.
            y: Y coordinate.
        """
        ...

    @abstractmethod
    def mouse_down(self, button: str = "left") -> None:
        """Press a mouse button down.

        Args:
            button: Button name ('left', 'right', 'middle').
        """
        ...

    @abstractmethod
    def mouse_up(self, button: str = "left") -> None:
        """Release a mouse button.

        Args:
            button: Button name.
        """
        ...

    @abstractmethod
    def mouse_click(self, x: int, y: int, button: str = "left") -> None:
        """Click at coordinates.

        This is a convenience method that moves, presses, and releases.

        Args:
            x: X coordinate.
            y: Y coordinate.
            button: Button name.
        """
        ...

    @abstractmethod
    def scroll(self, x: int, y: int, delta_x: int = 0, delta_y: int = 0) -> None:
        """Scroll at coordinates.

        Args:
            x: X coordinate.
            y: Y coordinate.
            delta_x: Horizontal scroll amount.
            delta_y: Vertical scroll amount (negative = up, positive = down).
        """
        ...


class NullInputBackend(InputBackend):
    """No-op backend for testing.

    This backend does nothing when input methods are called.
    It's used by controllers in unit tests where we don't want
    actual input to occur.

    All methods are no-ops that just log at debug level.
    """

    def key_down(self, key: str) -> None:
        """No-op key down."""
        logger.debug(f"NullInputBackend.key_down({key!r})")

    def key_up(self, key: str) -> None:
        """No-op key up."""
        logger.debug(f"NullInputBackend.key_up({key!r})")

    def type_char(self, char: str) -> None:
        """No-op type char."""
        logger.debug(f"NullInputBackend.type_char({char!r})")

    def mouse_move(self, x: int, y: int) -> None:
        """No-op mouse move."""
        logger.debug(f"NullInputBackend.mouse_move({x}, {y})")

    def mouse_down(self, button: str = "left") -> None:
        """No-op mouse down."""
        logger.debug(f"NullInputBackend.mouse_down({button!r})")

    def mouse_up(self, button: str = "left") -> None:
        """No-op mouse up."""
        logger.debug(f"NullInputBackend.mouse_up({button!r})")

    def mouse_click(self, x: int, y: int, button: str = "left") -> None:
        """No-op mouse click."""
        logger.debug(f"NullInputBackend.mouse_click({x}, {y}, {button!r})")

    def scroll(self, x: int, y: int, delta_x: int = 0, delta_y: int = 0) -> None:
        """No-op scroll."""
        logger.debug(f"NullInputBackend.scroll({x}, {y}, {delta_x}, {delta_y})")


# Mapping from common key names to Playwright key names
_PLAYWRIGHT_KEY_MAP: dict[str, str] = {
    # Modifiers
    "ctrl": "Control",
    "control": "Control",
    "alt": "Alt",
    "shift": "Shift",
    "meta": "Meta",
    "command": "Meta",
    "cmd": "Meta",
    "win": "Meta",
    "windows": "Meta",
    # Navigation
    "enter": "Enter",
    "return": "Enter",
    "tab": "Tab",
    "escape": "Escape",
    "esc": "Escape",
    "backspace": "Backspace",
    "delete": "Delete",
    "del": "Delete",
    "insert": "Insert",
    "home": "Home",
    "end": "End",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    # Arrows
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "arrowup": "ArrowUp",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    # Function keys
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "f5": "F5",
    "f6": "F6",
    "f7": "F7",
    "f8": "F8",
    "f9": "F9",
    "f10": "F10",
    "f11": "F11",
    "f12": "F12",
    # Other
    "space": " ",
    "spacebar": " ",
}


def _to_playwright_key(key: str) -> str:
    """Convert a key name to Playwright's expected format.

    Args:
        key: Key name in any common format.

    Returns:
        Key name in Playwright format.
    """
    key_lower = key.lower()
    return _PLAYWRIGHT_KEY_MAP.get(key_lower, key)


class PlaywrightInputBackend(InputBackend):
    """Backend that uses Playwright's page.keyboard and page.mouse.

    This backend sends actual input to a browser controlled by Playwright.
    It's used when the agent is running against a real browser.

    Example:
        >>> page = browser.new_page()
        >>> backend = PlaywrightInputBackend(page)
        >>> backend.type_char('a')  # Types 'a' in the browser
    """

    def __init__(self, page: Page) -> None:
        """Initialize with a Playwright page.

        Args:
            page: Playwright page object to send input to.
        """
        self._page = page
        logger.debug("PlaywrightInputBackend initialized")

    def key_down(self, key: str) -> None:
        """Press a key down using Playwright."""
        playwright_key = _to_playwright_key(key)
        logger.debug(f"PlaywrightInputBackend.key_down({key!r} -> {playwright_key!r})")
        self._page.keyboard.down(playwright_key)

    def key_up(self, key: str) -> None:
        """Release a key using Playwright."""
        playwright_key = _to_playwright_key(key)
        logger.debug(f"PlaywrightInputBackend.key_up({key!r} -> {playwright_key!r})")
        self._page.keyboard.up(playwright_key)

    def type_char(self, char: str) -> None:
        """Type a character using Playwright.

        For single characters, we use press() which handles the full
        key down/up cycle. For special characters that need shift,
        Playwright handles this automatically.
        """
        logger.debug(f"PlaywrightInputBackend.type_char({char!r})")
        # Use type() for actual character input as it handles unicode
        self._page.keyboard.type(char)

    def mouse_move(self, x: int, y: int) -> None:
        """Move mouse using Playwright."""
        logger.debug(f"PlaywrightInputBackend.mouse_move({x}, {y})")
        self._page.mouse.move(x, y)

    def mouse_down(self, button: str = "left") -> None:
        """Press mouse button using Playwright."""
        logger.debug(f"PlaywrightInputBackend.mouse_down({button!r})")
        self._page.mouse.down(button=button)

    def mouse_up(self, button: str = "left") -> None:
        """Release mouse button using Playwright."""
        logger.debug(f"PlaywrightInputBackend.mouse_up({button!r})")
        self._page.mouse.up(button=button)

    def mouse_click(self, x: int, y: int, button: str = "left") -> None:
        """Click at coordinates using Playwright."""
        logger.debug(f"PlaywrightInputBackend.mouse_click({x}, {y}, {button!r})")
        self._page.mouse.click(x, y, button=button)

    def scroll(self, x: int, y: int, delta_x: int = 0, delta_y: int = 0) -> None:
        """Scroll using Playwright.

        Playwright's mouse.wheel() scrolls at the current position,
        so we move first then scroll.
        """
        logger.debug(f"PlaywrightInputBackend.scroll({x}, {y}, {delta_x}, {delta_y})")
        self._page.mouse.move(x, y)
        self._page.mouse.wheel(delta_x, delta_y)
