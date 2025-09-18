import sqlite3, os, json, uuid
DB_PATH = os.getenv("DB_PATH","backend/db/app.db")

def log_provenance(session_id, agent, model_name, quantization, sources):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO provenance(prov_id,session_id,agent,model_name,quantization,sources) VALUES (?,?,?,?,?,?)",
                (uuid.uuid4().hex, session_id, agent, model_name, quantization, json.dumps(sources)))
    con.commit(); con.close()
