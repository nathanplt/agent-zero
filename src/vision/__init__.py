"""Vision system package for screen capture and game state extraction.

This package provides:
- ScreenshotCapture: Fast, reliable screenshot capture with buffering
- OCR: Text extraction (future)
- UI Detection: Element detection (future)
- LLM Vision: Complex understanding (future)
"""

from src.vision.capture import ScreenshotCapture

__all__ = ["ScreenshotCapture"]
