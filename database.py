import sqlite3
import uuid
from datetime import datetime, timezone

DB_PATH = "provenance_guard.db"

def now_utc():
    return datetime.now(timezone.utc).isoformat()


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS content (
                content_id TEXT PRIMARY KEY,
                creator_id TEXT NOT NULL,
                text TEXT NOT NULL,
                attribution TEXT NOT NULL,
                confidence REAL NOT NULL,
                combined_ai_score REAL NOT NULL,
                label TEXT NOT NULL,
                status TEXT NOT NULL,
                verified INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS appeals (
                appeal_id TEXT PRIMARY KEY,
                content_id TEXT NOT NULL,
                creator_reasoning TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (content_id) REFERENCES content(content_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                content_id TEXT,
                creator_id TEXT,
                timestamp TEXT NOT NULL,
                attribution TEXT,
                confidence REAL,
                combined_ai_score REAL,
                groq_score REAL,
                stylometric_score REAL,
                formulaic_score REAL,
                status TEXT,
                appeal_reasoning TEXT
            )
        """)

def save_content(content):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO content (
                content_id, creator_id, text, attribution, confidence,
                combined_ai_score, label, status, verified, created_at
            )
            VALUES (
                :content_id, :creator_id, :text, :attribution, :confidence,
                :combined_ai_score, :label, :status, :verified, :created_at
            )
        """, content)

def get_content(content_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM content WHERE content_id = ?",
            (content_id,)
        ).fetchone()

    return dict(row) if row else None

def update_content_status(content_id, status):
    with get_connection() as conn:
        conn.execute(
            "UPDATE content SET status = ? WHERE content_id = ?",
            (status, content_id)
        )

def verify_content(content_id):
    with get_connection() as conn:
        conn.execute(
            "UPDATE content SET verified = 1 WHERE content_id = ?",
            (content_id,)
        )

def save_appeal(content_id, creator_reasoning):
    appeal_id = str(uuid.uuid4())

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO appeals (
                appeal_id, content_id, creator_reasoning, created_at
            )
            VALUES (?, ?, ?, ?)
        """, (appeal_id, content_id, creator_reasoning, now_utc()))

    return appeal_id

def log_event(entry):
    entry = {
        "event_id": str(uuid.uuid4()),
        "event_type": entry.get("event_type"),
        "content_id": entry.get("content_id"),
        "creator_id": entry.get("creator_id"),
        "timestamp": now_utc(),
        "attribution": entry.get("attribution"),
        "confidence": entry.get("confidence"),
        "combined_ai_score": entry.get("combined_ai_score"),
        "groq_score": entry.get("groq_score"),
        "stylometric_score": entry.get("stylometric_score"),
        "formulaic_score": entry.get("formulaic_score"),
        "status": entry.get("status"),
        "appeal_reasoning": entry.get("appeal_reasoning"),
    }

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO audit_log (
                event_id, event_type, content_id, creator_id, timestamp,
                attribution, confidence, combined_ai_score,
                groq_score, stylometric_score, formulaic_score,
                status, appeal_reasoning
            )
            VALUES (
                :event_id, :event_type, :content_id, :creator_id, :timestamp,
                :attribution, :confidence, :combined_ai_score,
                :groq_score, :stylometric_score, :formulaic_score,
                :status, :appeal_reasoning
            )
        """, entry)

def read_log(limit=20):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                audit_log.*,
                content.text AS submitted_text
            FROM audit_log
            LEFT JOIN content
                ON audit_log.content_id = content.content_id
            ORDER BY audit_log.timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [dict(row) for row in rows]

def get_analytics():
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS count FROM content").fetchone()["count"]

        if total == 0:
            return {
                "total_submissions": 0,
                "likely_ai_ratio": 0,
                "likely_human_ratio": 0,
                "uncertain_ratio": 0,
                "appeal_rate": 0,
                "average_confidence": 0,
            }

        counts = conn.execute("""
            SELECT attribution, COUNT(*) AS count
            FROM content
            GROUP BY attribution
        """).fetchall()

        appeal_count = conn.execute("SELECT COUNT(*) AS count FROM appeals").fetchone()["count"]

        avg_confidence = conn.execute("""
            SELECT AVG(confidence) AS average
            FROM content
        """).fetchone()["average"]

    attribution_counts = {row["attribution"]: row["count"] for row in counts}

    return {
        "total_submissions": total,
        "likely_ai_ratio": round(attribution_counts.get("likely_ai", 0) / total, 2),
        "likely_human_ratio": round(attribution_counts.get("likely_human", 0) / total, 2),
        "uncertain_ratio": round(attribution_counts.get("uncertain", 0) / total, 2),
        "appeal_rate": round(appeal_count / total, 2),
        "average_confidence": round(avg_confidence or 0, 2),
    }

def get_latest_classification_log(content_id):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT groq_score, stylometric_score, formulaic_score
            FROM audit_log
            WHERE content_id = ?
              AND event_type IN ('classification', 'metadata_classification')
            ORDER BY timestamp DESC
            LIMIT 1
        """, (content_id,)).fetchone()

    return dict(row) if row else {}
