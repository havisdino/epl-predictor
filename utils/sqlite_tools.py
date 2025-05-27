import sqlite3
import json


def load_jsonl_to_sqlite(jsonl_path, db_path="data/epl.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            date TEXT,
            team TEXT,
            opponent TEXT,
            venue TEXT,
            result TEXT,
            goals_for INTEGER,
            goals_against INTEGER
        )
    """)

    # Create index for performance
    c.execute("CREATE INDEX IF NOT EXISTS idx_team_date ON matches(team, date)")

    # Insert data
    with open(jsonl_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            c.execute("""
                INSERT INTO matches (date, team, opponent, venue, result, goals_for, goals_against)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                rec["date"],
                rec["team"],
                rec["opponent"],
                rec["venue"],
                rec["result"],
                rec["goals_for"],
                rec["goals_against"]
            ))
    conn.commit()
    conn.close()


def get_n_previous_matches_sqlite(date_str, N, db_path="data/epl.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Query previous matches
    c.execute("""
        SELECT date, venue, result, team, opponent, goals_for, goals_against
        FROM matches
        WHERE date < ?
        ORDER BY date DESC
        LIMIT ?
    """, (date_str, N))

    rows = c.fetchall()
    conn.close()

    # Convert to dicts for usability
    return [
        {
            "date": row[0],
            "venue": row[1],
            "result": row[2],
            "team": row[3],
            "opponent": row[4],
            "goals_for": row[5],
            "goals_against": row[6]
        }
        for row in rows
    ]


def update_sqlite_with_new_records(jsonl_path, db_path="data/epl.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Ensure table and index exist
    c.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            date TEXT,
            team TEXT,
            opponent TEXT,
            venue TEXT,
            result TEXT,
            goals_for INTEGER,
            goals_against INTEGER,
            PRIMARY KEY (date, team, opponent)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_team_date ON matches(team, date)")

    # Load new records
    with open(jsonl_path, "r") as f:
        new_records = [json.loads(line) for line in f]

    # Insert records
    for rec in new_records:
        c.execute("""
            INSERT OR IGNORE INTO matches (date, team, opponent, venue, result, goals_for, goals_against)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            rec["date"],
            rec["team"],
            rec["opponent"],
            rec["venue"],
            rec["result"],
            rec["goals_for"],
            rec["goals_against"]
        ))

    conn.commit()
    conn.close()