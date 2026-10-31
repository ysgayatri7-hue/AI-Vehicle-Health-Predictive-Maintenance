"""Small SQLite persistence layer for prediction history."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import DATABASE_PATH


def connect(path=DATABASE_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def initialize(path=DATABASE_PATH):
    with connect(path) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            failure_probability REAL NOT NULL,
            anomaly_prediction INTEGER NOT NULL,
            source TEXT NOT NULL DEFAULT 'web'
        )""")


def save_prediction(result, source="web", path=DATABASE_PATH):
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect(path) as db:
        cursor = db.execute(
            "INSERT INTO predictions (created_at, inputs_json, result_json, prediction, failure_probability, anomaly_prediction, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (created_at, json.dumps(result["inputs"]), json.dumps(result), result["prediction"],
             result["failure_probability"], result["anomaly_prediction"], source),
        )
        return cursor.lastrowid


def get_prediction(prediction_id, path=DATABASE_PATH):
    with connect(path) as db:
        row = db.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    return _decode(row) if row else None


def list_predictions(limit=200, path=DATABASE_PATH):
    with connect(path) as db:
        rows = db.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    return [_decode(row) for row in rows]


def _decode(row):
    item = dict(row)
    item["inputs"] = json.loads(item.pop("inputs_json"))
    item["result"] = json.loads(item.pop("result_json"))
    return item
