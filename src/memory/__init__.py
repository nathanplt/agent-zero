"""Memory package for state persistence and learning.

This package provides:
- SQLiteStatePersistence: Persistent storage for game states using SQLite
- SQLiteEpisodicMemory: Episodic memory for action-outcome pairs
- PersistenceError: Base error for persistence operations
- CorruptedDataError: Error for data corruption detection
- EpisodicMemoryError: Error for episodic memory operations
"""

from src.memory.episodic import EpisodicMemoryError, SQLiteEpisodicMemory
from src.memory.persistence import (
    CorruptedDataError,
    PersistenceError,
    SQLiteStatePersistence,
)

__all__ = [
    "CorruptedDataError",
    "EpisodicMemoryError",
    "PersistenceError",
    "SQLiteEpisodicMemory",
    "SQLiteStatePersistence",
]
