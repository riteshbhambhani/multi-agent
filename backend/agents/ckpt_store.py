import sqlite3, os, uuid, json
DB_PATH = os.getenv("DB_PATH","backend/db/app.db")

def create(user_id, session_id, pending_agent, pending_question, context_snapshot):
    con = sqlite3.connect(DB_PATH)
    ckpt_id = uuid.uuid4().hex
    con.execute("INSERT INTO checkpoints(checkpoint_id,user_id,session_id,pending_agent,pending_question,context_snapshot) VALUES (?,?,?,?,?,?)",
                (ckpt_id, user_id, session_id, pending_agent, pending_question, context_snapshot))
    con.commit(); con.close()
    return {"checkpoint_id": ckpt_id}

def get(ckpt_id):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT checkpoint_id,user_id,session_id,pending_agent,pending_question,context_snapshot FROM checkpoints WHERE checkpoint_id=?",(ckpt_id,)).fetchone()
    con.close()
    if not row: return None
    keys = ["checkpoint_id","user_id","session_id","pending_agent","pending_question","context_snapshot"]
    return dict(zip(keys,row))

def delete(ckpt_id):
    con = sqlite3.connect(DB_PATH); con.execute("DELETE FROM checkpoints WHERE checkpoint_id=?",(ckpt_id,)); con.commit(); con.close()
