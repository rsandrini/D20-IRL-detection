import sqlite3
import json
import secrets
from datetime import datetime, timezone

DB_PATH = "dice.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                enabled     INTEGER DEFAULT 1,
                daily_limit INTEGER DEFAULT 0,
                token       TEXT,
                created_at  TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rolls (
                id                    TEXT PRIMARY KEY,
                user_id               TEXT,
                mode                  TEXT,
                detections            TEXT,
                selected_face         INTEGER,
                image_path            TEXT,
                gif_path              TEXT,
                time_elapsed          REAL,
                time_elapsed_detection REAL,
                created_at            TEXT,
                reported_wrong        INTEGER DEFAULT 0,
                correct_face          INTEGER
            )
        """)
        conn.execute("""
            INSERT OR IGNORE INTO users (id, enabled, daily_limit, token, created_at)
            VALUES ('local', 1, 0, NULL, ?)
        """, (_now(),))
        conn.commit()


def _now():
    return datetime.now(timezone.utc).isoformat()


def save_roll(roll_id, user_id, mode, detections, selected_face,
              image_path, gif_path, time_elapsed, time_elapsed_detection):
    with _connect() as conn:
        conn.execute("""
            INSERT INTO rolls (id, user_id, mode, detections, selected_face,
                image_path, gif_path, time_elapsed, time_elapsed_detection, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (roll_id, user_id, mode, json.dumps(detections), selected_face,
              image_path, gif_path, time_elapsed, time_elapsed_detection, _now()))
        conn.commit()


def report_roll(roll_id, correct_face):
    with _connect() as conn:
        conn.execute(
            "UPDATE rolls SET reported_wrong=1, correct_face=? WHERE id=?",
            (correct_face, roll_id)
        )
        conn.commit()
        return conn.execute("SELECT changes()").fetchone()[0]


def get_user(user_id):
    with _connect() as conn:
        return conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()


def get_user_by_token(token):
    with _connect() as conn:
        return conn.execute("SELECT * FROM users WHERE token=?", (token,)).fetchone()


def list_users():
    with _connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM users ORDER BY created_at"
        ).fetchall()]


def create_user(user_id, daily_limit=0):
    token = secrets.token_hex(16)
    with _connect() as conn:
        conn.execute("""
            INSERT INTO users (id, enabled, daily_limit, token, created_at)
            VALUES (?, 1, ?, ?, ?)
        """, (user_id, daily_limit, token, _now()))
        conn.commit()
    return token


def update_user(user_id, **kwargs):
    allowed = {"enabled", "daily_limit"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return False
    set_clause = ", ".join(f"{k}=?" for k in fields)
    with _connect() as conn:
        conn.execute(
            f"UPDATE users SET {set_clause} WHERE id=?",
            (*fields.values(), user_id)
        )
        conn.commit()
    return True


def get_user_roll_count_today(user_id):
    today = datetime.now(timezone.utc).date().isoformat()
    with _connect() as conn:
        row = conn.execute("""
            SELECT COUNT(*) as cnt FROM rolls WHERE user_id=? AND created_at LIKE ?
        """, (user_id, f"{today}%")).fetchone()
        return row["cnt"]


def get_rolls(user_id=None, limit=50):
    with _connect() as conn:
        if user_id:
            rows = conn.execute("""
                SELECT * FROM rolls WHERE user_id=? ORDER BY created_at DESC LIMIT ?
            """, (user_id, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM rolls ORDER BY created_at DESC LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_roll(roll_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM rolls WHERE id=?", (roll_id,)).fetchone()
        return dict(row) if row else None
