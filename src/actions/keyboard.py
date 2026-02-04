"""Keyboard control module for human-like keyboard input.

This module provides:
- Text typing with natural timing variance
- Special key support (Enter, Escape, Tab, etc.)
- Key combinations (Ctrl+A, Ctrl+C, etc.)
- Key down/up control for advanced use cases

Example:
    >>> from src.actions.keyboard import KeyboardController
    >>>
    >>> controller = KeyboardController()
    >>> controller.type_text("Hello World")
    >>> controller.press_key("enter")
    >>> controller.key_combo(["ctrl", "a"])
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Key aliases for common alternative names
KEY_ALIASES: dict[str, str] = {
    "return": "enter",
    "esc": "escape",
    "control": "ctrl",
    "cmd": "command",
    "win": "windows",
    "del": "delete",
    "bs": "backspace",
    "space": "space",
    "spacebar": "space",
}

# Valid special keys
SPECIAL_KEYS: set[str] = {
    # Modifier keys
    "ctrl",
    "alt",
    "shift",
    "command",
    "windows",
    # Navigation keys
    "enter",
    "escape",
    "tab",
    "backspace",
    "delete",
    "insert",
    "home",
    "end",
    "pageup",
    "pagedown",
    # Arrow keys
    "up",
    "down",
    "left",
    "right",
    # Function keys
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    "f12",
    # Other
    "space",
    "capslock",
    "numlock",
    "scrolllock",
    "printscreen",
    "pause",
}


def calculate_typing_delay(
    base_ms: float = 80.0,
    variance_ms: float = 40.0,
) -> float:
    """Calculate delay between keystrokes with natural variance.

    Simulates human typing speed which varies naturally between
    keystrokes.

    Args:
        base_ms: Base delay in milliseconds.
        variance_ms: Standard deviation of variance.

    Returns:
        Delay in seconds.
    """
    delay_ms = base_ms + random.gauss(0, variance_ms)
    # Clamp to reasonable range (20ms to 200ms)
    delay_ms = max(20, min(200, delay_ms))
    return delay_ms / 1000.0


def calculate_key_press_duration(
    base_ms: float = 50.0,
    variance_ms: float = 20.0,
) -> float:
    """Calculate how long a key is held down.

    Args:
        base_ms: Base duration in milliseconds.
        variance_ms: Standard deviation of variance.

    Returns:
        Duration in seconds.
    """
    duration_ms = base_ms + random.gauss(0, variance_ms)
    # Clamp to reasonable range
    duration_ms = max(20, min(150, duration_ms))
    return duration_ms / 1000.0


def normalize_key(key: str) -> str:
    """Normalize a key name to standard form.

    Handles case normalization and aliases.

    Args:
        key: Key name to normalize.

    Returns:
        Normalized key name.
    """
    key_lower = key.lower()
    return KEY_ALIASES.get(key_lower, key_lower)


class KeyboardController:
    """Controller for human-like keyboard input.

    Provides natural typing with timing variance,
    special key support, and key combinations.

    Attributes:
        _base_delay_ms: Base delay between keystrokes.
        _delay_variance_ms: Variance in keystroke timing.

    Example:
        >>> controller = KeyboardController()
        >>> controller.type_text("Hello")
        >>> controller.press_key("enter")
        >>> controller.key_combo(["ctrl", "c"])
    """

    def __init__(
        self,
        base_delay_ms: float = 80.0,
        delay_variance_ms: float = 40.0,
    ) -> None:
        """Initialize the keyboard controller.

        Args:
            base_delay_ms: Base delay between keystrokes in milliseconds.
            delay_variance_ms: Variance in keystroke timing.
        """
        self._base_delay_ms = base_delay_ms
        self._delay_variance_ms = delay_variance_ms

        logger.debug(
            f"KeyboardController initialized with base_delay={base_delay_ms}ms, "
            f"variance={delay_variance_ms}ms"
        )

    def _press_char(self, char: str) -> None:
        """Press a single character key.

        This is the internal method that performs the actual key press.
        Override or mock for testing.

        Args:
            char: Character to type.
        """
        # Default implementation does nothing
        # Real implementation would use pyautogui or similar
        logger.debug(f"Press char: {char!r}")

    def _key_down(self, key: str) -> None:
        """Press a key down (without releasing).

        Args:
            key: Key name.
        """
        # Default implementation does nothing
        logger.debug(f"Key down: {key}")

    def _key_up(self, key: str) -> None:
        """Release a key.

        Args:
            key: Key name.
        """
        # Default implementation does nothing
        logger.debug(f"Key up: {key}")

    def type_text(
        self,
        text: str,
        interval_ms: float | None = None,
    ) -> None:
        """Type text with natural timing.

        Types each character with human-like delays between
        keystrokes.

        Args:
            text: Text to type.
            interval_ms: Optional fixed interval between keystrokes.
                If None, uses natural variance.
        """
        if not text:
            return

        for i, char in enumerate(text):
            self._press_char(char)

            # Add delay between characters (not after the last one)
            if i < len(text) - 1:
                if interval_ms is not None:
                    delay = interval_ms / 1000.0
                else:
                    delay = calculate_typing_delay(
                        self._base_delay_ms,
                        self._delay_variance_ms,
                    )
                time.sleep(delay)

        logger.debug(f"Typed text: {text!r}")

    def press_key(self, key: str) -> None:
        """Press and release a single key.

        Handles special keys like Enter, Escape, Tab, etc.

        Args:
            key: Key to press (e.g., 'enter', 'escape', 'tab').
        """
        normalized = normalize_key(key)

        # Press and release with natural timing
        duration = calculate_key_press_duration()

        self._key_down(normalized)
        time.sleep(duration)
        self._key_up(normalized)

        logger.debug(f"Pressed key: {normalized}")

    def key_combo(self, keys: list[str]) -> None:
        """Press a key combination.

        Presses all keys in order, then releases in reverse order.
        This is how key combinations work (e.g., Ctrl+A: hold Ctrl,
        press A, release A, release Ctrl).

        Args:
            keys: List of keys to press together (e.g., ['ctrl', 'a']).
        """
        if not keys:
            return

        # Normalize all keys
        normalized_keys = [normalize_key(k) for k in keys]

        # Press all keys in order
        for key in normalized_keys:
            self._key_down(key)
            time.sleep(calculate_key_press_duration() * 0.3)  # Short delay between presses

        # Small delay while holding
        time.sleep(calculate_key_press_duration() * 0.5)

        # Release in reverse order
        for key in reversed(normalized_keys):
            self._key_up(key)
            time.sleep(calculate_key_press_duration() * 0.2)

        logger.debug(f"Key combo: {'+'.join(normalized_keys)}")

    def key_down(self, key: str) -> None:
        """Press a key down without releasing.

        Useful for holding modifier keys or advanced input.

        Args:
            key: Key to press down.
        """
        normalized = normalize_key(key)
        self._key_down(normalized)
        logger.debug(f"Key down: {normalized}")

    def key_up(self, key: str) -> None:
        """Release a key.

        Args:
            key: Key to release.
        """
        normalized = normalize_key(key)
        self._key_up(normalized)
        logger.debug(f"Key up: {normalized}")
