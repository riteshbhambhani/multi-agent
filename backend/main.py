import os, json, uuid, asyncio
from fastapi import FastAPI, WebSocket, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import sqlite3

from models.model_loader import load_llm
from agents.benefit import BenefitAgent
from agents.claim import ClaimAgent
from agents.summary import SummaryAgent
from agents import ckpt_store
from agents.orchestrator import build_graph, GraphState

load_dotenv()
DB_PATH = os.getenv("DB_PATH","backend/db/app.db")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# global singletons
LLM = load_llm()
benefit_agent = BenefitAgent(LLM)
claim_agent = ClaimAgent(LLM)
summary_agent = SummaryAgent(LLM)
graph = build_graph(benefit_agent, claim_agent, summary_agent, ckpt_store)

class ChatSend(BaseModel):
    session_id: str
    user_id: str
    text: str

@app.post("/api/session/create")
def session_create(user_id: Optional[str]=Form(None), title: Optional[str]=Form(None)):
    con = sqlite3.connect(DB_PATH)
    if not user_id: user_id = "u" + uuid.uuid4().hex[:6]
    session_id = "s_" + uuid.uuid4().hex[:8]
    con.execute("INSERT OR REPLACE INTO users(user_id) VALUES (?)",(user_id,))
    con.execute("INSERT INTO sessions(session_id,user_id,title) VALUES (?,?,?)",(session_id,user_id,title or "New Chat"))
    con.commit(); con.close()
    return {"session_id": session_id, "user_id": user_id}

@app.post("/api/chat/send")
def chat_send(req: ChatSend):
    token = uuid.uuid4().hex
    store = {"session_id": req.session_id, "user_id": req.user_id, "text": req.text, "token": token}
    _pending_streams[token] = store
    return {"stream_url": f"/api/stream/{req.session_id}/{token}"}

@app.post("/api/chat/resume")
def chat_resume(checkpoint_id: str = Form(...), text: str = Form(...)):
    ck = ckpt_store.get(checkpoint_id)
    if not ck: return {"error":"invalid_checkpoint"}
    token = uuid.uuid4().hex
    _pending_streams[token] = {"resume": True, "ckpt": ck, "text": text, "token": token}
    return {"stream_url": f"/api/stream/{ck['session_id']}/{token}"}

@app.post("/api/files/ingest")
async def ingest_files(benefits: UploadFile = File(None), claims: UploadFile = File(None)):
    os.makedirs("backend/data", exist_ok=True)
    out = {}
    if benefits:
        bpath = "backend/data/benefits.json"; out["benefits_path"]=bpath
        with open(bpath,"wb") as f: f.write(await benefits.read())
    if claims:
        cpath = "backend/data/claims.json"; out["claims_path"]=cpath
        with open(cpath,"wb") as f: f.write(await claims.read())
    return out

@app.get("/api/provenance/{session_id}")
def get_prov(session_id: str):
    con = sqlite3.connect(DB_PATH)
    rows = [dict(session_id=session_id, agent=a, model_name=b, quantization=c, sources=json.loads(d)) 
            for a,b,c,d in con.execute("SELECT agent,model_name,quantization,sources FROM provenance WHERE session_id=?",(session_id,))]
    con.close(); return rows

@app.get("/api/checkpoints/{session_id}")
def list_ckpts(session_id: str):
    con = sqlite3.connect(DB_PATH)
    out = []
    for row in con.execute("SELECT checkpoint_id,pending_agent,pending_question,created_at FROM checkpoints WHERE session_id=?",(session_id,)):
        out.append({"checkpoint_id":row[0],"pending_agent":row[1],"pending_question":row[2],"created_at":row[3]})
    con.close(); return out

_pending_streams = {}

@app.websocket("/api/stream/{session_id}/{token}")
async def ws_stream(ws: WebSocket, session_id: str, token: str):
    await ws.accept()
    payload = _pending_streams.pop(token, None)
    if not payload:
        await ws.send_json({"type":"error","data":"no_pending"}); await ws.close(); return

    # persist user message
    con = sqlite3.connect(DB_PATH)
    if "resume" in payload:
        state_json = json.loads(payload["ckpt"]["context_snapshot"])
        state_json["question"] = payload["text"]
        ckpt_store.delete(payload["ckpt"]["checkpoint_id"])
        state = GraphState(**state_json)
    else:
        con.execute("INSERT INTO messages(message_id,session_id,role,content,agent) VALUES (?,?,?,?,?)",
                    (uuid.uuid4().hex, payload["session_id"], "user", payload["text"], "user"))
        con.commit()
        state = GraphState(session_id=payload["session_id"], user_id=payload["user_id"], question=payload["text"])

    try:
        for ev in graph.stream(state):
            # ev is intermediate GraphState updates
            if isinstance(ev, GraphState): 
                await ws.send_json({"type":"token","data":""})
        final = graph.invoke(state)
        # persist assistant message
        con.execute("INSERT INTO messages(message_id,session_id,role,content,agent) VALUES (?,?,?,?,?)",
                    (uuid.uuid4().hex, session_id, "assistant", (final.summary or ""), "summary"))
        con.commit(); con.close()

        await ws.send_json({"type":"meta","data":{"agent":"summary","provenance":final.provenance,"checkpoint_id":final.checkpoint_id}})
        await ws.send_json({"type":"done"})
    except Exception as e:
        await ws.send_json({"type":"error","data":str(e)})
        await ws.close()
