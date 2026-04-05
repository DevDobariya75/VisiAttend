"""
database.py — Neon.tech PostgreSQL + pgvector connection helper
===============================================================
Used by attendance_scanner.py via:
    from database import get_connection, register_vector

Set the DATABASE_URL environment variable to your Neon.tech connection string:
    export DATABASE_URL="postgresql://user:password@host/dbname?sslmode=require"

Required packages:
    pip install psycopg2-binary pgvector
"""

import os
from pathlib import Path

import psycopg2
from pgvector.psycopg2 import register_vector as _register_vector

# ── Connection string ─────────────────────────────────────────────────────────
# Reads from environment variable. Set this in your shell or .env file:
#   DATABASE_URL=postgresql://<user>:<password>@<host>/<db>?sslmode=require
def _load_database_url() -> str | None:
    """Load DATABASE_URL from environment, then fallback to project .env."""
    env_value = os.environ.get("DATABASE_URL")
    if env_value:
        return env_value

    # backend/database.py -> project root: ../
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return None

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "DATABASE_URL":
                return value.strip().strip('"').strip("'")
    except OSError:
        return None

    return None


DATABASE_URL = _load_database_url()

if not DATABASE_URL:
    raise EnvironmentError(
        "[database.py] DATABASE_URL environment variable is not set.\n"
        "Set it before running scripts:\n"
        "  Windows CMD: set DATABASE_URL=postgresql://user:pass@host/db?sslmode=require\n"
        "  PowerShell : $env:DATABASE_URL='postgresql://user:pass@host/db?sslmode=require'\n"
        "Or create project .env with: DATABASE_URL=postgresql://..."
    )


def get_connection() -> psycopg2.extensions.connection:
    """
    Open and return a new psycopg2 connection to Neon.tech.
    Callers are responsible for closing the connection after use.
    
    Neon.tech requires SSL — the ?sslmode=require in DATABASE_URL handles this.
    """
    return psycopg2.connect(DATABASE_URL)


def register_vector(conn: psycopg2.extensions.connection) -> None:
    """
    Register the pgvector type adapter for a given connection.
    Must be called once per connection before executing vector queries.
    """
    _register_vector(conn)


# ── Optional: DB schema bootstrap ─────────────────────────────────────────────
# Run this once to create tables on a fresh Neon.tech database.
SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

-- Register table: one row per embedding sample for a student roll number.
CREATE TABLE IF NOT EXISTS register (
    id        SERIAL PRIMARY KEY,
    roll_no   TEXT NOT NULL,
    name      TEXT NOT NULL,
    embedding vector(512) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS register_embedding_idx
    ON register USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS register_roll_no_idx
    ON register (roll_no);

-- One attendance session (slot) created by teacher.
CREATE TABLE IF NOT EXISTS attendance_slots (
    id            SERIAL PRIMARY KEY,
    class_id      TEXT NOT NULL UNIQUE,
    subject_name  TEXT NOT NULL,
    lecture_room  TEXT NOT NULL,
    faculty_name  TEXT NOT NULL,
    start_time    TIMESTAMPTZ NOT NULL,
    end_time      TIMESTAMPTZ NOT NULL,
    status        TEXT NOT NULL DEFAULT 'open'
                  CHECK (status IN ('open', 'completed')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at  TIMESTAMPTZ
);

-- Per-student status for a specific slot.
CREATE TABLE IF NOT EXISTS attendance_slot_records (
    id         SERIAL PRIMARY KEY,
    slot_id    INTEGER NOT NULL REFERENCES attendance_slots(id) ON DELETE CASCADE,
    class_id   TEXT NOT NULL,
    roll_no    TEXT NOT NULL,
    name       TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'absent'
               CHECK (status IN ('present', 'absent')),
    marked_at  TIMESTAMPTZ,
    UNIQUE (slot_id, roll_no)
);

CREATE INDEX IF NOT EXISTS attendance_slot_records_slot_idx
    ON attendance_slot_records (slot_id);

-- Migration-safe updates for old schema without class_id columns.
ALTER TABLE attendance_slots
    ADD COLUMN IF NOT EXISTS class_id TEXT;

UPDATE attendance_slots
SET class_id = CONCAT('CLASS-', id)
WHERE class_id IS NULL;

ALTER TABLE attendance_slots
    ALTER COLUMN class_id SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS attendance_slots_class_id_uidx
    ON attendance_slots (class_id);

ALTER TABLE attendance_slot_records
    ADD COLUMN IF NOT EXISTS class_id TEXT;

UPDATE attendance_slot_records r
SET class_id = s.class_id
FROM attendance_slots s
WHERE r.slot_id = s.id
  AND r.class_id IS NULL;

ALTER TABLE attendance_slot_records
    ALTER COLUMN class_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS attendance_slot_records_class_id_idx
    ON attendance_slot_records (class_id);
"""

def bootstrap_schema():
    """Create tables and indexes if they don't exist. Safe to run multiple times."""
    conn = get_connection()
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()
    conn.close()
    print("[database.py] Schema bootstrap complete.")


if __name__ == "__main__":
    bootstrap_schema()