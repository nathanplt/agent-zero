"""Memory package for state persistence and learning.

This package provides:
- SQLiteStatePersistence: Persistent storage for game states using SQLite
- PersistenceError: Base error for persistence operations
- CorruptedDataError: Error for data corruption detection
"""

from src.memory.persistence import (
    CorruptedDataError,
    PersistenceError,
    SQLiteStatePersistence,
)

__all__ = [
    "CorruptedDataError",
    "PersistenceError",
    "SQLiteStatePersistence",
]
