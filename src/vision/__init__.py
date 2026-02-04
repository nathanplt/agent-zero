"""Vision system package for screen capture and game state extraction.

This package provides:
- ScreenshotCapture: Fast, reliable screenshot capture with buffering
- OCRSystem: Text extraction using OCR with number parsing
- UI Detection: Element detection (future)
- LLMVision: LLM-based vision for complex game state understanding
"""

from src.vision.capture import ScreenshotCapture
from src.vision.llm_vision import LLMVision, LLMVisionError, add_set_of_mark
from src.vision.ocr import OCRSystem

__all__ = [
    "ScreenshotCapture",
    "OCRSystem",
    "LLMVision",
    "LLMVisionError",
    "add_set_of_mark",
]
