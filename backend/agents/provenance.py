import sqlite3, os, json, uuid, logging
DB_PATH = os.getenv("DB_PATH","backend/db/app.db")
logger = logging.getLogger("backend.provenance")

def log_provenance(session_id, agent, model_name, quantization, sources):
    logger.info("Logging provenance session=%s agent=%s model=%s sources=%d", session_id, agent, model_name, len(sources))
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO provenance(prov_id,session_id,agent,model_name,quantization,sources) VALUES (?,?,?,?,?,?)",
                (uuid.uuid4().hex, session_id, agent, model_name, quantization, json.dumps(sources)))
    con.commit(); con.close()
