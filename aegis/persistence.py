"""SQLite persistence for AEGIS training, imports, and predictions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis.config import DEFAULT_DB_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AegisRepository:
    """Small SQLite repository so the demo behaves like a real backend."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    rows INTEGER NOT NULL,
                    fraud_rows INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prediction_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    transaction_id TEXT NOT NULL,
                    invoice_id TEXT NOT NULL,
                    final_score REAL NOT NULL,
                    risk_band TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dataset_imports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    input_paths_json TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    label_summary_json TEXT NOT NULL,
                    notes TEXT NOT NULL
                );
                """
            )

    def record_training_run(self, summary: dict[str, Any], source_name: str = "synthetic") -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO training_runs (created_at, source_name, model_path, rows, fraud_rows, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_now(),
                    source_name,
                    str(summary["model_path"]),
                    int(summary["rows"]),
                    int(summary["fraud_rows"]),
                    json.dumps(summary["metrics"]),
                ),
            )

    def record_prediction(self, request_payload: dict[str, Any], response_payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO prediction_events (
                    created_at,
                    transaction_id,
                    invoice_id,
                    final_score,
                    risk_band,
                    request_json,
                    response_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_now(),
                    str(response_payload["transaction_id"]),
                    str(response_payload["invoice_id"]),
                    float(response_payload["final_score"]),
                    str(response_payload["risk_band"]),
                    json.dumps(request_payload, default=str),
                    json.dumps(response_payload, default=str),
                ),
            )

    def record_dataset_import(
        self,
        source_name: str,
        input_paths: list[str],
        output_path: str,
        row_count: int,
        label_summary: dict[str, int],
        notes: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO dataset_imports (
                    created_at,
                    source_name,
                    input_paths_json,
                    output_path,
                    row_count,
                    label_summary_json,
                    notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_now(),
                    source_name,
                    json.dumps(input_paths),
                    output_path,
                    int(row_count),
                    json.dumps(label_summary),
                    notes,
                ),
            )

    def list_training_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM training_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_dict(row, json_fields={"metrics_json": "metrics"}) for row in rows]

    def list_prediction_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM prediction_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            self._row_to_dict(
                row,
                json_fields={
                    "request_json": "request",
                    "response_json": "response",
                },
            )
            for row in rows
        ]

    def list_dataset_imports(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM dataset_imports
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            self._row_to_dict(
                row,
                json_fields={
                    "input_paths_json": "input_paths",
                    "label_summary_json": "label_summary",
                },
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row, json_fields: dict[str, str]) -> dict[str, Any]:
        payload = dict(row)
        for source_key, target_key in json_fields.items():
            payload[target_key] = json.loads(payload.pop(source_key))
        return payload
