"""Environment management package.

This package provides implementations for managing the execution environment:
- BrowserRuntime: Manages browser lifecycle via Playwright
"""

from src.environment.browser import BrowserRuntime

__all__ = ["BrowserRuntime"]
