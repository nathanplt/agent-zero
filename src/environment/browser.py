"""Browser runtime management using Playwright.

This module provides a BrowserRuntime class that manages the Chromium browser
lifecycle via Playwright. It enables:
- Launching Chromium in headed or headless mode
- Navigating to URLs
- Taking screenshots
- Managing browser lifecycle
"""

from __future__ import annotations

import contextlib
import io
import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from PIL import Image
    from playwright.sync_api import Browser, BrowserContext, Page, Playwright

logger = logging.getLogger(__name__)


class BrowserRuntimeError(Exception):
    """Error raised when browser operations fail."""

    pass


class BrowserRuntime:
    """Manages browser lifecycle via Playwright.

    This class provides methods to:
    - Start/stop the browser
    - Navigate to URLs
    - Take screenshots
    - Access the underlying Playwright page for advanced operations

    Example:
        >>> runtime = BrowserRuntime(headless=False)
        >>> runtime.start()
        >>> runtime.navigate("https://example.com")
        >>> screenshot = runtime.screenshot()
        >>> runtime.stop()
    """

    def __init__(
        self,
        headless: bool = False,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
        user_agent: str | None = None,
    ) -> None:
        """Initialize browser runtime.

        Args:
            headless: Whether to run browser in headless mode.
                      Set to False to use the virtual display (Xvfb).
            viewport_width: Browser viewport width in pixels.
            viewport_height: Browser viewport height in pixels.
            user_agent: Custom user agent string. If None, uses default.
        """
        self._headless = headless
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._user_agent = user_agent

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Check if browser is running."""
        return self._is_running and self._browser is not None

    @property
    def page(self) -> Page | None:
        """Get the current page object for advanced operations."""
        return self._page

    def start(self) -> None:
        """Start the browser.

        Raises:
            BrowserRuntimeError: If browser fails to start.
        """
        if self._is_running:
            logger.warning("Browser is already running")
            return

        try:
            from playwright.sync_api import sync_playwright

            logger.info("Starting Playwright...")
            self._playwright = sync_playwright().start()

            logger.info(f"Launching Chromium (headless={self._headless})...")
            self._browser = self._playwright.chromium.launch(
                headless=self._headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )

            # Create context with viewport settings
            context_options: dict[str, Any] = {
                "viewport": {
                    "width": self._viewport_width,
                    "height": self._viewport_height,
                },
            }
            if self._user_agent:
                context_options["user_agent"] = self._user_agent

            self._context = self._browser.new_context(**context_options)
            self._page = self._context.new_page()
            self._is_running = True

            logger.info(
                f"Browser started successfully "
                f"(viewport: {self._viewport_width}x{self._viewport_height})"
            )

        except ImportError as e:
            raise BrowserRuntimeError(
                "Playwright not installed. Install with: pip install playwright"
            ) from e
        except Exception as e:
            self._cleanup()
            raise BrowserRuntimeError(f"Failed to start browser: {e}") from e

    def stop(self) -> None:
        """Stop the browser and cleanup resources."""
        logger.info("Stopping browser...")
        self._cleanup()
        logger.info("Browser stopped")

    def _cleanup(self) -> None:
        """Clean up browser resources."""
        if self._page:
            with contextlib.suppress(Exception):
                self._page.close()
            self._page = None

        if self._context:
            with contextlib.suppress(Exception):
                self._context.close()
            self._context = None

        if self._browser:
            with contextlib.suppress(Exception):
                self._browser.close()
            self._browser = None

        if self._playwright:
            with contextlib.suppress(Exception):
                self._playwright.stop()
            self._playwright = None

        self._is_running = False

    def navigate(self, url: str, timeout_ms: int = 30000, wait_until: str = "load") -> None:
        """Navigate to a URL.

        Args:
            url: The URL to navigate to.
            timeout_ms: Navigation timeout in milliseconds.
            wait_until: When to consider navigation complete.
                       Options: "load", "domcontentloaded", "networkidle", "commit"

        Raises:
            BrowserRuntimeError: If navigation fails or browser not running.
        """
        if not self._page:
            raise BrowserRuntimeError("Browser not running. Call start() first.")

        try:
            logger.info(f"Navigating to: {url}")
            self._page.goto(url, timeout=timeout_ms, wait_until=cast(Any, wait_until))
            logger.info(f"Navigation complete. Title: {self._page.title()}")
        except Exception as e:
            raise BrowserRuntimeError(f"Navigation to {url} failed: {e}") from e

    def screenshot(self, full_page: bool = False) -> bytes:
        """Take a screenshot of the current page.

        Args:
            full_page: If True, capture the full scrollable page.
                      If False, capture only the viewport.

        Returns:
            PNG-encoded screenshot bytes.

        Raises:
            BrowserRuntimeError: If screenshot fails or browser not running.
        """
        if not self._page:
            raise BrowserRuntimeError("Browser not running. Call start() first.")

        try:
            result: bytes = self._page.screenshot(full_page=full_page)
            return result
        except Exception as e:
            raise BrowserRuntimeError(f"Screenshot failed: {e}") from e

    def screenshot_pil(self, full_page: bool = False) -> Image.Image:
        """Take a screenshot and return as PIL Image.

        Args:
            full_page: If True, capture the full scrollable page.

        Returns:
            PIL Image object.

        Raises:
            BrowserRuntimeError: If screenshot fails.
        """
        from PIL import Image

        screenshot_bytes = self.screenshot(full_page=full_page)
        return Image.open(io.BytesIO(screenshot_bytes))

    def get_page_title(self) -> str:
        """Get the current page title.

        Returns:
            The page title string.

        Raises:
            BrowserRuntimeError: If browser not running.
        """
        if not self._page:
            raise BrowserRuntimeError("Browser not running. Call start() first.")
        title: str = self._page.title()
        return title

    def get_page_url(self) -> str:
        """Get the current page URL.

        Returns:
            The current URL string.

        Raises:
            BrowserRuntimeError: If browser not running.
        """
        if not self._page:
            raise BrowserRuntimeError("Browser not running. Call start() first.")
        url: str = self._page.url
        return url

    def get_page_content(self) -> str:
        """Get the current page HTML content.

        Returns:
            The full HTML content of the page.

        Raises:
            BrowserRuntimeError: If browser not running.
        """
        if not self._page:
            raise BrowserRuntimeError("Browser not running. Call start() first.")
        content: str = self._page.content()
        return content

    def get_viewport_size(self) -> tuple[int, int]:
        """Get the configured viewport size.

        Returns:
            Tuple of (width, height) in pixels.
        """
        return (self._viewport_width, self._viewport_height)

    def __enter__(self) -> BrowserRuntime:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """Context manager exit."""
        self.stop()

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self._cleanup()
