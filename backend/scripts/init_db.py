import sqlite3, os, pathlib, logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("backend.scripts.init_db")
DB_PATH = os.getenv("DB_PATH","backend/db/app.db")
pathlib.Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

logger.info("Initializing DB at %s", DB_PATH)
sql = pathlib.Path("backend/schemas/sql.sql").read_text()
con = sqlite3.connect(DB_PATH)
con.executescript(sql)
con.commit()
con.close()
logger.info("DB initialized at %s", DB_PATH)
print("DB initialized at", DB_PATH)
