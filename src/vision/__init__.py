"""Vision system package for screen capture and game state extraction.

This package provides:
- ScreenshotCapture: Fast, reliable screenshot capture with buffering
- OCRSystem: Text extraction using OCR with number parsing
- UI Detection: Element detection (future)
- LLM Vision: Complex understanding (future)
"""

from src.vision.capture import ScreenshotCapture
from src.vision.ocr import OCRSystem

__all__ = ["ScreenshotCapture", "OCRSystem"]
