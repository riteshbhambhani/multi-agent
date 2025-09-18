import sqlite3, os, pathlib
from dotenv import load_dotenv
load_dotenv()

DB_PATH = os.getenv("DB_PATH","backend/db/app.db")
pathlib.Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

sql = pathlib.Path("backend/schemas/sql.sql").read_text()
con = sqlite3.connect(DB_PATH)
con.executescript(sql)
con.commit()
con.close()
print("DB initialized at", DB_PATH)
