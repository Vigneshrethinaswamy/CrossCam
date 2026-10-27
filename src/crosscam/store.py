"""Small SQLite metadata store for tracklets and association hypotheses."""
import json
import sqlite3
from pathlib import Path
from typing import Self

from .models import AssociationResult, Tracklet


class TrackStore:
    def __init__(self, path: str | Path = "data/crosscam.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tracklets (
                tracklet_id TEXT PRIMARY KEY,
                camera_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                embedding_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS associations (
                source_tracklet_id TEXT NOT NULL,
                candidate_tracklet_id TEXT NOT NULL,
                score REAL NOT NULL,
                decision TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                PRIMARY KEY (source_tracklet_id, candidate_tracklet_id)
            );
            """
        )
        self._connection.commit()

    def upsert_tracklet(self, tracklet: Tracklet) -> None:
        self._connection.execute(
            """INSERT OR REPLACE INTO tracklets
               (tracklet_id, camera_id, start_time, end_time, embedding_json, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (tracklet.tracklet_id, tracklet.camera_id, tracklet.start_time, tracklet.end_time,
             json.dumps(tracklet.embedding), json.dumps(tracklet.metadata)),
        )
        self._connection.commit()

    def save_association(self, result: AssociationResult) -> None:
        self._connection.execute(
            """INSERT OR REPLACE INTO associations
               (source_tracklet_id, candidate_tracklet_id, score, decision, reasons_json)
               VALUES (?, ?, ?, ?, ?)""",
            (result.source_tracklet_id, result.candidate_tracklet_id, result.score,
             result.decision.value, json.dumps(result.reasons)),
        )
        self._connection.commit()

    def list_tracklets(self) -> list[dict]:
        rows = self._connection.execute(
            "SELECT tracklet_id, camera_id, start_time, end_time FROM tracklets ORDER BY start_time"
        ).fetchall()
        return [
            {"tracklet_id": row[0], "camera_id": row[1], "start_time": row[2], "end_time": row[3]}
            for row in rows
        ]

    def list_associations(self) -> list[dict]:
        rows = self._connection.execute(
            "SELECT source_tracklet_id, candidate_tracklet_id, score, decision, reasons_json "
            "FROM associations ORDER BY score DESC"
        ).fetchall()
        return [
            {"source_tracklet_id": row[0], "candidate_tracklet_id": row[1], "score": row[2],
             "decision": row[3], "reasons": json.loads(row[4])}
            for row in rows
        ]

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
