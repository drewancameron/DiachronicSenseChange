#!/usr/bin/env python3
"""Initialize the SQLite database from schema.sql."""

import sqlite3
import sys
from pathlib import Path

DB_DIR = Path(__file__).parent.parent / "db"
SCHEMA_PATH = DB_DIR / "schema.sql"
DB_PATH = DB_DIR / "diachronic.db"


def init_db(db_path: Path = DB_PATH, schema_path: Path = SCHEMA_PATH) -> None:
    if not schema_path.exists():
        print(f"Schema file not found: {schema_path}", file=sys.stderr)
        sys.exit(1)

    schema_sql = schema_path.read_text()
    conn = sqlite3.connect(db_path)
    conn.executescript(schema_sql)
    conn.close()
    print(f"Database initialized at {db_path}")


if __name__ == "__main__":
    init_db()
