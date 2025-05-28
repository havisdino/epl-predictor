import sys
from utils.sqlite_tools import recompute_team_stats, update_sqlite_with_new_records


for jsonl_file in sys.argv[1:]:
    print(f"Updating SQLite database with records from {jsonl_file}")
    update_sqlite_with_new_records(jsonl_file, db_path="data/epl.db")
    recompute_team_stats(db_path="data/epl.db")