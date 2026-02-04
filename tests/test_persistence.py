"""Tests for the SQLiteStatePersistence class."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from src.memory.persistence import (
    CorruptedDataError,
    PersistenceError,
    SQLiteStatePersistence,
)


class TestSQLiteStatePersistenceInit:
    """Tests for SQLiteStatePersistence initialization."""

    def test_creates_with_default_path(self, tmp_path: Path) -> None:
        """Test initialization with default path."""
        db_path = tmp_path / "test.db"
        persistence = SQLiteStatePersistence(db_path)
        assert persistence.db_path == db_path
        persistence.close()

    def test_creates_database_file(self, tmp_path: Path) -> None:
        """Test that database file is created."""
        db_path = tmp_path / "test.db"
        persistence = SQLiteStatePersistence(db_path)
        assert db_path.exists()
        persistence.close()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test that parent directories are created."""
        db_path = tmp_path / "subdir" / "nested" / "test.db"
        persistence = SQLiteStatePersistence(db_path)
        assert db_path.parent.exists()
        persistence.close()

    def test_initializes_schema(self, tmp_path: Path) -> None:
        """Test that schema is initialized."""
        db_path = tmp_path / "test.db"
        persistence = SQLiteStatePersistence(db_path)

        # Verify table exists
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_states'"
        )
        assert cursor.fetchone() is not None
        conn.close()
        persistence.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test context manager usage."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            persistence.save_state({"test": 1})
        # Should not raise after exit


class TestSaveState:
    """Tests for save_state method."""

    def test_save_simple_state(self, tmp_path: Path) -> None:
        """Test saving a simple state."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            state_id = persistence.save_state({"gold": 100})
            assert state_id > 0

    def test_save_complex_state(self, tmp_path: Path) -> None:
        """Test saving a complex nested state."""
        db_path = tmp_path / "test.db"
        state = {
            "resources": {"gold": 1000, "gems": 50},
            "level": 10,
            "inventory": [{"id": 1, "name": "sword"}, {"id": 2, "name": "shield"}],
            "settings": {"volume": 0.8, "fullscreen": True},
        }
        with SQLiteStatePersistence(db_path) as persistence:
            state_id = persistence.save_state(state)
            assert state_id > 0

    def test_save_with_metadata(self, tmp_path: Path) -> None:
        """Test saving state with metadata."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            state_id = persistence.save_state(
                {"gold": 100},
                metadata={"source": "manual_save", "version": "1.0"},
            )
            assert state_id > 0

    def test_save_increments_id(self, tmp_path: Path) -> None:
        """Test that save increments state ID."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            id1 = persistence.save_state({"a": 1})
            id2 = persistence.save_state({"b": 2})
            id3 = persistence.save_state({"c": 3})
            assert id2 > id1
            assert id3 > id2

    def test_save_non_serializable_raises(self, tmp_path: Path) -> None:
        """Test that non-serializable state raises error."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            # Circular reference is not JSON serializable
            circular: dict[str, Any] = {}
            circular["self"] = circular
            with pytest.raises(PersistenceError):
                persistence.save_state(circular)


class TestLoadState:
    """Tests for load_state method."""

    def test_load_returns_none_when_empty(self, tmp_path: Path) -> None:
        """Test loading from empty database."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            assert persistence.load_state() is None

    def test_load_returns_most_recent(self, tmp_path: Path) -> None:
        """Test that load returns most recent state."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            persistence.save_state({"version": 1})
            persistence.save_state({"version": 2})
            persistence.save_state({"version": 3})

            loaded = persistence.load_state()
            assert loaded is not None
            assert loaded["version"] == 3

    def test_load_verifies_checksum_by_default(self, tmp_path: Path) -> None:
        """Test that checksum is verified by default."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            persistence.save_state({"gold": 100})

        # Corrupt the data directly
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE game_states SET state_json = '{\"gold\": 999}' WHERE id = 1"
        )
        conn.commit()
        conn.close()

        with (
            SQLiteStatePersistence(db_path) as persistence,
            pytest.raises(CorruptedDataError),
        ):
            persistence.load_state()

    def test_load_can_skip_checksum(self, tmp_path: Path) -> None:
        """Test loading without checksum verification."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            persistence.save_state({"gold": 100})

        # Corrupt the data directly
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE game_states SET state_json = '{\"gold\": 999}' WHERE id = 1"
        )
        conn.commit()
        conn.close()

        with SQLiteStatePersistence(db_path) as persistence:
            loaded = persistence.load_state(verify_checksum=False)
            assert loaded is not None
            assert loaded["gold"] == 999

    def test_save_then_load_roundtrip(self, tmp_path: Path) -> None:
        """Test save and load roundtrip preserves data."""
        db_path = tmp_path / "test.db"
        original = {
            "resources": {"gold": 1000, "gems": 50},
            "level": 10,
            "active": True,
        }

        with SQLiteStatePersistence(db_path) as persistence:
            persistence.save_state(original)
            loaded = persistence.load_state()

        assert loaded == original


class TestLoadStateById:
    """Tests for load_state_by_id method."""

    def test_load_by_id_returns_correct_state(self, tmp_path: Path) -> None:
        """Test loading specific state by ID."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            id1 = persistence.save_state({"version": 1})
            id2 = persistence.save_state({"version": 2})
            id3 = persistence.save_state({"version": 3})

            state1 = persistence.load_state_by_id(id1)
            state2 = persistence.load_state_by_id(id2)
            state3 = persistence.load_state_by_id(id3)

            assert state1 is not None and state1["version"] == 1
            assert state2 is not None and state2["version"] == 2
            assert state3 is not None and state3["version"] == 3

    def test_load_by_id_returns_none_for_missing(self, tmp_path: Path) -> None:
        """Test loading non-existent state returns None."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            persistence.save_state({"test": 1})
            assert persistence.load_state_by_id(9999) is None


class TestGetStateHistory:
    """Tests for get_state_history method."""

    def test_history_returns_empty_list_when_empty(self, tmp_path: Path) -> None:
        """Test history returns empty list for empty database."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            history = persistence.get_state_history()
            assert history == []

    def test_history_returns_states_in_order(self, tmp_path: Path) -> None:
        """Test history returns states most recent first."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            for i in range(5):
                persistence.save_state({"index": i})

            history = persistence.get_state_history(limit=5)
            assert len(history) == 5
            assert history[0]["index"] == 4  # Most recent
            assert history[4]["index"] == 0  # Oldest

    def test_history_respects_limit(self, tmp_path: Path) -> None:
        """Test history respects limit parameter."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            for i in range(20):
                persistence.save_state({"index": i})

            history = persistence.get_state_history(limit=5)
            assert len(history) == 5

    def test_history_filters_by_since(self, tmp_path: Path) -> None:
        """Test history filters by since parameter."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            # Save some states
            for i in range(5):
                persistence.save_state({"index": i})

            # Get states after a recent time
            since = datetime.now() - timedelta(seconds=1)
            history = persistence.get_state_history(since=since)

            # Should get all 5 states (saved within last second)
            assert len(history) == 5

    def test_history_1000_states_query_last_10(self, tmp_path: Path) -> None:
        """Test querying last 10 from 1000 states."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            for i in range(1000):
                persistence.save_state({"index": i})

            history = persistence.get_state_history(limit=10)
            assert len(history) == 10
            # Most recent should be index 999
            assert history[0]["index"] == 999


class TestStateCount:
    """Tests for get_state_count method."""

    def test_count_empty_database(self, tmp_path: Path) -> None:
        """Test count on empty database."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            assert persistence.get_state_count() == 0

    def test_count_after_saves(self, tmp_path: Path) -> None:
        """Test count after saving states."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            for i in range(10):
                persistence.save_state({"index": i})
            assert persistence.get_state_count() == 10


class TestDeleteState:
    """Tests for delete_state method."""

    def test_delete_existing_state(self, tmp_path: Path) -> None:
        """Test deleting an existing state."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            state_id = persistence.save_state({"test": 1})
            assert persistence.delete_state(state_id) is True
            assert persistence.load_state_by_id(state_id) is None

    def test_delete_nonexistent_state(self, tmp_path: Path) -> None:
        """Test deleting non-existent state returns False."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            assert persistence.delete_state(9999) is False


class TestPruneOldStates:
    """Tests for prune_old_states method."""

    def test_prune_keeps_recent(self, tmp_path: Path) -> None:
        """Test pruning keeps most recent states."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            for i in range(20):
                persistence.save_state({"index": i})

            deleted = persistence.prune_old_states(keep_count=10)
            assert deleted == 10
            assert persistence.get_state_count() == 10

            # Verify most recent are kept
            history = persistence.get_state_history(limit=10)
            indices = [s["index"] for s in history]
            assert 19 in indices  # Most recent kept
            assert 0 not in indices  # Oldest pruned

    def test_prune_nothing_to_prune(self, tmp_path: Path) -> None:
        """Test pruning when nothing to prune."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            for i in range(5):
                persistence.save_state({"index": i})

            deleted = persistence.prune_old_states(keep_count=10)
            assert deleted == 0
            assert persistence.get_state_count() == 5


class TestClearAll:
    """Tests for clear_all method."""

    def test_clear_all_removes_everything(self, tmp_path: Path) -> None:
        """Test clear_all removes all states."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            for i in range(10):
                persistence.save_state({"index": i})

            deleted = persistence.clear_all()
            assert deleted == 10
            assert persistence.get_state_count() == 0


class TestVerifyIntegrity:
    """Tests for verify_integrity method."""

    def test_verify_all_valid(self, tmp_path: Path) -> None:
        """Test verification with all valid states."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            for i in range(10):
                persistence.save_state({"index": i})

            valid, corrupted = persistence.verify_integrity()
            assert valid == 10
            assert corrupted == 0

    def test_verify_detects_corruption(self, tmp_path: Path) -> None:
        """Test verification detects corrupted states."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            for i in range(5):
                persistence.save_state({"index": i})

        # Corrupt some data directly
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE game_states SET state_json = '{\"corrupted\": true}' WHERE id IN (1, 3)"
        )
        conn.commit()
        conn.close()

        with SQLiteStatePersistence(db_path) as persistence:
            valid, corrupted = persistence.verify_integrity()
            assert valid == 3
            assert corrupted == 2


class TestRepairDatabase:
    """Tests for repair_database method."""

    def test_repair_removes_corrupted(self, tmp_path: Path) -> None:
        """Test repair removes corrupted entries."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            for i in range(5):
                persistence.save_state({"index": i})

        # Corrupt some data directly
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE game_states SET state_json = '{\"corrupted\": true}' WHERE id IN (1, 3)"
        )
        conn.commit()
        conn.close()

        with SQLiteStatePersistence(db_path) as persistence:
            removed = persistence.repair_database()
            assert removed == 2
            assert persistence.get_state_count() == 3

            # Verify remaining are valid
            valid, corrupted = persistence.verify_integrity()
            assert valid == 3
            assert corrupted == 0


class TestExportImport:
    """Tests for export_to_json and import_from_json methods."""

    def test_export_to_json(self, tmp_path: Path) -> None:
        """Test exporting states to JSON file."""
        db_path = tmp_path / "test.db"
        export_path = tmp_path / "export.json"

        with SQLiteStatePersistence(db_path) as persistence:
            for i in range(5):
                persistence.save_state({"index": i})

            exported = persistence.export_to_json(export_path)
            assert exported == 5

        # Verify file contents
        with open(export_path) as f:
            data = json.load(f)
        assert len(data) == 5

    def test_import_from_json(self, tmp_path: Path) -> None:
        """Test importing states from JSON file."""
        db_path = tmp_path / "test.db"
        import_path = tmp_path / "import.json"

        # Create import file
        states = [{"index": i} for i in range(5)]
        with open(import_path, "w") as f:
            json.dump(states, f)

        with SQLiteStatePersistence(db_path) as persistence:
            imported = persistence.import_from_json(import_path)
            assert imported == 5
            assert persistence.get_state_count() == 5

    def test_export_import_roundtrip(self, tmp_path: Path) -> None:
        """Test export then import preserves data."""
        db_path1 = tmp_path / "test1.db"
        db_path2 = tmp_path / "test2.db"
        export_path = tmp_path / "export.json"

        # Save states to first database
        with SQLiteStatePersistence(db_path1) as persistence:
            for i in range(5):
                persistence.save_state({"index": i, "data": f"item_{i}"})
            persistence.export_to_json(export_path)

        # Import to second database
        with SQLiteStatePersistence(db_path2) as persistence:
            persistence.import_from_json(export_path)
            history = persistence.get_state_history(limit=10)

        # Verify all states present
        assert len(history) == 5
        indices = {s["index"] for s in history}
        assert indices == {0, 1, 2, 3, 4}


class TestCorruptedDataHandling:
    """Tests for graceful handling of corrupted data."""

    def test_corrupted_json_raises_error(self, tmp_path: Path) -> None:
        """Test that corrupted JSON raises CorruptedDataError."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            persistence.save_state({"test": 1})

        # Corrupt JSON directly
        conn = sqlite3.connect(str(db_path))
        # Update checksum too so it "passes" but JSON is invalid
        invalid_json = "not valid json {"
        checksum = SQLiteStatePersistence._compute_checksum(invalid_json)
        conn.execute(
            "UPDATE game_states SET state_json = ?, checksum = ?",
            (invalid_json, checksum),
        )
        conn.commit()
        conn.close()

        with (
            SQLiteStatePersistence(db_path) as persistence,
            pytest.raises(CorruptedDataError),
        ):
            persistence.load_state()

    def test_history_skips_corrupted_json(self, tmp_path: Path) -> None:
        """Test that history skips entries with corrupted JSON."""
        db_path = tmp_path / "test.db"
        with SQLiteStatePersistence(db_path) as persistence:
            persistence.save_state({"index": 1})
            persistence.save_state({"index": 2})
            persistence.save_state({"index": 3})

        # Corrupt one entry's JSON
        conn = sqlite3.connect(str(db_path))
        invalid_json = "not valid json {"
        checksum = SQLiteStatePersistence._compute_checksum(invalid_json)
        conn.execute(
            "UPDATE game_states SET state_json = ?, checksum = ? WHERE id = 2",
            (invalid_json, checksum),
        )
        conn.commit()
        conn.close()

        with SQLiteStatePersistence(db_path) as persistence:
            history = persistence.get_state_history(limit=10)
            # Should have 2 valid entries, skipping corrupted one
            assert len(history) == 2


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_saves(self, tmp_path: Path) -> None:
        """Test concurrent save operations."""
        db_path = tmp_path / "test.db"
        errors: list[Exception] = []

        def save_thread(persistence: SQLiteStatePersistence, thread_id: int) -> None:
            try:
                for i in range(50):
                    persistence.save_state({"thread": thread_id, "index": i})
            except Exception as e:
                errors.append(e)

        with SQLiteStatePersistence(db_path) as persistence:
            threads = [
                threading.Thread(target=save_thread, args=(persistence, i))
                for i in range(5)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            # 5 threads * 50 saves = 250 states
            assert persistence.get_state_count() == 250


class TestModuleExports:
    """Tests for module exports."""

    def test_exports_from_package(self) -> None:
        """Test that classes can be imported from package."""
        from src.memory import (
            CorruptedDataError,
            PersistenceError,
            SQLiteStatePersistence,
        )

        assert SQLiteStatePersistence is not None
        assert PersistenceError is not None
        assert CorruptedDataError is not None

    def test_persistence_error_is_exception(self) -> None:
        """Test that PersistenceError is an Exception."""
        error = PersistenceError("test")
        assert isinstance(error, Exception)

    def test_corrupted_data_error_is_persistence_error(self) -> None:
        """Test that CorruptedDataError inherits from PersistenceError."""
        error = CorruptedDataError("test")
        assert isinstance(error, PersistenceError)
