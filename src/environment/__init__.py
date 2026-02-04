"""Environment management package.

This package provides implementations for managing the execution environment:
- LocalEnvironmentManager: Coordinates display and browser lifecycle
- VirtualDisplay: Manages virtual X11 display (Xvfb)
- BrowserRuntime: Manages browser lifecycle via Playwright
"""

from src.environment.browser import BrowserRuntime
from src.environment.display import VirtualDisplay
from src.environment.manager import LocalEnvironmentManager

__all__ = ["BrowserRuntime", "LocalEnvironmentManager", "VirtualDisplay"]
