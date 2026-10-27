"""Local persistence for aggregate attendance and engagement sessions."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from io import StringIO
from typing import Dict, Optional
from uuid import uuid4

import pandas as pd

from analytics_engine import generate_session_analytics


def default_database_path() -> Path:
    """Return an OS-local path that is not stored alongside project files."""
    if os.name == "nt":
        base_directory = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base_directory = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base_directory / "MultimodelAttendance" / "session_history.sqlite3"


def _initialize(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            audio_source TEXT NOT NULL,
            tracked_profiles INTEGER NOT NULL,
            present_profiles INTEGER NOT NULL,
            attendance_rate REAL NOT NULL,
            expected_participants INTEGER,
            roster_attendance_rate REAL,
            mean_engagement REAL NOT NULL,
            minimum_engagement REAL NOT NULL,
            metrics_json TEXT NOT NULL,
            attendance_json TEXT NOT NULL
        )
        """
    )
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
    }
    if "expected_participants" not in columns:
        connection.execute("ALTER TABLE sessions ADD COLUMN expected_participants INTEGER")
    if "roster_attendance_rate" not in columns:
        connection.execute("ALTER TABLE sessions ADD COLUMN roster_attendance_rate REAL")


def save_session(
    metrics: pd.DataFrame,
    attendance: pd.DataFrame,
    audio_source: str,
    started_at: Optional[str] = None,
    expected_participants: Optional[int] = None,
    database_path: Optional[Path] = None,
) -> str:
    """Save aggregate session data and return its opaque local identifier."""
    path = database_path or default_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = generate_session_analytics(
        attendance, metrics, expected_participants=expected_participants
    )
    session_id = str(uuid4())
    observed_at = started_at or datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            _initialize(connection)
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, started_at, audio_source, tracked_profiles,
                    present_profiles, attendance_rate, expected_participants,
                    roster_attendance_rate, mean_engagement, minimum_engagement,
                    metrics_json, attendance_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    observed_at,
                    audio_source,
                    summary["tracked_profiles"],
                    summary["present_profiles"],
                    summary["track_confirmation_rate"],
                    expected_participants,
                    summary["attendance_rate"],
                    summary["mean_engagement"],
                    summary["minimum_engagement"],
                    metrics.to_json(orient="records", date_format="iso"),
                    attendance.to_json(orient="records", date_format="iso"),
                ),
            )
            connection.execute(
                """
                DELETE FROM sessions WHERE session_id IN (
                    SELECT session_id FROM sessions
                    ORDER BY started_at DESC LIMIT -1 OFFSET 50
                )
                """
            )
    return session_id


def list_sessions(database_path: Optional[Path] = None) -> list[Dict[str, object]]:
    """Return the most recent session summaries without loading metric history."""
    path = database_path or default_database_path()
    if not path.is_file():
        return []
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        with connection:
            _initialize(connection)
            rows = connection.execute(
                """
                SELECT session_id, started_at, audio_source, tracked_profiles,
                      present_profiles, attendance_rate AS track_confirmation_rate,
                      expected_participants, roster_attendance_rate,
                      mean_engagement, minimum_engagement
                FROM sessions ORDER BY started_at DESC LIMIT 50
                """
            ).fetchall()
    return [dict(row) for row in rows]


def load_session(session_id: str, database_path: Optional[Path] = None) -> Optional[Dict[str, object]]:
    """Load one saved session and reconstruct its metrics and attendance tables."""
    path = database_path or default_database_path()
    if not path.is_file():
        return None
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        with connection:
            _initialize(connection)
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
    if row is None:
        return None
    record = dict(row)
    record["metrics"] = pd.read_json(StringIO(record.pop("metrics_json")), orient="records")
    record["attendance"] = pd.read_json(StringIO(record.pop("attendance_json")), orient="records")
    return record


def update_session_roster(
    session_id: str,
    expected_participants: int,
    database_path: Optional[Path] = None,
) -> bool:
    """Update a saved session's expected class size and roster attendance."""
    if expected_participants < 1:
        raise ValueError("expected_participants must be positive.")
    path = database_path or default_database_path()
    if not path.is_file():
        return False
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            _initialize(connection)
            cursor = connection.execute(
                """
                UPDATE sessions
                SET expected_participants = ?,
                    roster_attendance_rate = present_profiles * 100.0 / ?
                WHERE session_id = ?
                """,
                (expected_participants, expected_participants, session_id),
            )
            return cursor.rowcount == 1


def clear_history(database_path: Optional[Path] = None) -> int:
    """Delete saved session data and return the number of removed sessions."""
    path = database_path or default_database_path()
    if not path.is_file():
        return 0
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            _initialize(connection)
            cursor = connection.execute("DELETE FROM sessions")
            return cursor.rowcount