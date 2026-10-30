"""Persistent SQLite storage for ML battle results."""

import json
import sqlite3

from datetime import datetime, timezone
from pathlib import Path


# Store the database inside the project's instance directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "instance"
DATABASE_PATH = DATABASE_DIR / "battle_history.db"


def get_connection():
    """Create a database connection."""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=10
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_history_db():
    """Create the battle history table if it doesn't exist."""

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS battles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                dataset TEXT NOT NULL,
                task_type TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                feature_count INTEGER NOT NULL,
                ranking_metric TEXT NOT NULL,
                winner_name TEXT,
                winner_algorithm_key TEXT,
                winner_score REAL,
                results_json TEXT NOT NULL
            )
            """
        )


def save_battle(result):
    """Save a completed battle and return its database ID."""

    if not isinstance(result, dict):
        raise ValueError("Battle result must be a dictionary.")

    required_fields = (
        "dataset",
        "task_type",
        "sample_count",
        "feature_count",
        "ranking_metric",
        "results",
    )

    missing_fields = [
        field for field in required_fields
        if field not in result
    ]

    if missing_fields:
        raise ValueError(
            f"Missing battle result fields: {', '.join(missing_fields)}"
        )

    winner = result.get("winner") or {}

    # Store the complete result so metrics can be reviewed later.
    serialized_result = json.dumps(
        result,
        allow_nan=False
    )

    created_at = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )

    initialize_history_db()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO battles (
                created_at,
                dataset,
                task_type,
                sample_count,
                feature_count,
                ranking_metric,
                winner_name,
                winner_algorithm_key,
                winner_score,
                results_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                str(result["dataset"]),
                str(result["task_type"]),
                int(result["sample_count"]),
                int(result["feature_count"]),
                str(result["ranking_metric"]),
                winner.get("name"),
                winner.get("algorithm_key"),
                winner.get("score"),
                serialized_result,
            ),
        )

        return cursor.lastrowid


def get_recent_battles(limit=10):
    """Return the latest saved battles, newest first."""

    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("Limit must be an integer.")

    if not 1 <= limit <= 100:
        raise ValueError("Limit must be between 1 and 100.")

    initialize_history_db()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                dataset,
                task_type,
                sample_count,
                feature_count,
                ranking_metric,
                winner_name,
                winner_algorithm_key,
                winner_score,
                results_json
            FROM battles
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    battles = []

    for row in rows:
        battle = dict(row)

        # Convert the stored JSON into usable Python data.
        battle["results"] = json.loads(
            battle.pop("results_json")
        )

        battles.append(battle)

    return battles