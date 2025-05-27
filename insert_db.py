import sqlite3
import json


jsonl_file = 'data/raw/matches.jsonl'
conn = sqlite3.connect('epl_matches.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    venue TEXT,
    result TEXT,
    team TEXT,
    opponent TEXT,
    goals_for INTEGER,
    goals_against INTEGER
)
''')

with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        cursor.execute('''
            INSERT INTO matches (date, venue, result, team, opponent, goals_for, goals_against)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['date'],
            data['venue'],
            data['result'],
            data['team'],
            data['opponent'],
            data['goals_for'],
            data['goals_against']
        ))

conn.commit()
conn.close()

print("Data inserted into 'epl_matches.db' successfully.")