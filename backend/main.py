import os, json, uuid, asyncio, logging
# avoid tokenizers parallelism warning when forking (silences spurious messages)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from contextvars import ContextVar
from fastapi import FastAPI, WebSocket, UploadFile, Form, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import sqlite3

from .models.model_loader import load_llm
from .agents.benefit import BenefitAgent
from .agents.claim import ClaimAgent
from .agents.summary import SummaryAgent
from .agents import ckpt_store
from .agents.orchestrator import build_graph, GraphState
from .agents.orchestrator import SemanticRouter

load_dotenv()
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# logging
log_file = os.path.join(LOG_DIR, "app.log")
REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="-")

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        try:
            record.request_id = REQUEST_ID_CTX.get()
        except Exception:
            record.request_id = "-"
        return True

formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(request_id)s] - %(message)s")
fh = logging.FileHandler(log_file)
fh.setFormatter(formatter)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
fh.addFilter(RequestIdFilter())
sh.addFilter(RequestIdFilter())
root = logging.getLogger()
root.setLevel(logging.INFO)
root.addHandler(fh)
root.addHandler(sh)
logger = logging.getLogger("backend")
logger.info("Starting backend application")
DB_PATH = os.getenv("DB_PATH","backend/db/app.db")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def add_request_id(request, call_next):
    # prefer incoming request id if present; otherwise generate one
    rid = request.headers.get("x-request-id") or ("r-" + uuid.uuid4().hex[:8])
    token = REQUEST_ID_CTX.set(rid)
    try:
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response
    finally:
        REQUEST_ID_CTX.reset(token)

# global singletons (initialized at startup)
LLM = None
benefit_agent = None
claim_agent = None
summary_agent = None
graph = None


@app.on_event("startup")
async def startup_event():
    """Initialize heavy models and agents once FastAPI starts (avoids import-time side-effects).
    """
    global LLM, benefit_agent, claim_agent, summary_agent, graph
    # load LLM and agents
    logger.info("Initializing LLM and agents")
    LLM = load_llm()
    benefit_agent = BenefitAgent(LLM)
    claim_agent = ClaimAgent(LLM)
    summary_agent = SummaryAgent(LLM)
    graph = build_graph(benefit_agent, claim_agent, summary_agent, ckpt_store)
    logger.info("Agents initialized and graph built")
    # warm-up semantic router to avoid first-request latency (optional)
    try:
        from .agents import orchestrator
        orchestrator.router_node._sr = SemanticRouter()
        logger.info("SemanticRouter warmed up at startup")
    except Exception as e:
        logger.warning("SemanticRouter warm-up failed: %s", e)

class ChatSend(BaseModel):
    session_id: str
    user_id: str
    text: str

@app.post("/api/session/create")
async def session_create(request: Request, user_id: Optional[str]=Form(None), title: Optional[str]=Form(None)):
    """Create a new chat session.

    This endpoint accepts either multipart form-data (from browser FormData) or
    application/json. Returning both allows the frontend to POST an empty JSON
    body without triggering the python-multipart parser warning when the
    request contains no fields.
    """
    # If the client sent JSON, prefer values from the JSON body.
    try:
        ct = request.headers.get("content-type", "")
        if ct.startswith("application/json"):
            body = await request.json()
            if isinstance(body, dict):
                user_id = body.get("user_id") or user_id
                title = body.get("title") or title
    except Exception:
        # If JSON parsing fails, fall back to form values already provided.
        pass

    logger.info("API /api/session/create called by content-type=%s", request.headers.get("content-type"))
    con = sqlite3.connect(DB_PATH)
    if not user_id: user_id = "u" + uuid.uuid4().hex[:6]
    session_id = "s_" + uuid.uuid4().hex[:8]
    con.execute("INSERT OR REPLACE INTO users(user_id) VALUES (?)",(user_id,))
    con.execute("INSERT INTO sessions(session_id,user_id,title) VALUES (?,?,?)",(session_id,user_id,title or "New Chat"))
    con.commit(); con.close()
    return {"session_id": session_id, "user_id": user_id}

@app.post("/api/chat/send")
def chat_send(req: ChatSend):
    logger.info("API /api/chat/send session=%s user=%s", req.session_id, req.user_id)
    token = uuid.uuid4().hex
    store = {"session_id": req.session_id, "user_id": req.user_id, "text": req.text, "token": token}
    _pending_streams[token] = store
    return {"stream_url": f"/api/stream/{req.session_id}/{token}"}

@app.post("/api/chat/resume")
def chat_resume(checkpoint_id: str = Form(...), text: str = Form(...)):
    logger.info("API /api/chat/resume checkpoint=%s", checkpoint_id)
    ck = ckpt_store.get(checkpoint_id)
    if not ck: return {"error":"invalid_checkpoint"}
    token = uuid.uuid4().hex
    _pending_streams[token] = {"resume": True, "ckpt": ck, "text": text, "token": token}
    return {"stream_url": f"/api/stream/{ck['session_id']}/{token}"}

@app.post("/api/files/ingest")
async def ingest_files(benefits: UploadFile = File(None), claims: UploadFile = File(None)):
    logger.info("API /api/files/ingest called benefits=%s claims=%s", getattr(benefits, 'filename', None), getattr(claims, 'filename', None))
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
        final_state = None
        for ev in graph.stream(state):
            # If the stream yields a GraphState, keep it as the latest final state.
            if isinstance(ev, GraphState):
                final_state = ev
            else:
                # ev is a token chunk (string) — forward it to the client
                await ws.send_json({"type":"token","data": ev})
        # If the streaming returned a final GraphState or dict-like state, use it; otherwise invoke once.
        if final_state is not None:
            final = final_state
        else:
            final = graph.invoke(state)

        # Normalize final to GraphState model if possible (graph.stream may yield dict-like states)
        try:
            if not isinstance(final, GraphState):
                # try to coerce dict-like final into GraphState
                if isinstance(final, dict):
                    final = GraphState(**final)
                else:
                    # try mapping-like interface
                    try:
                        final = GraphState(**dict(final))
                    except Exception:
                        logger.warning("Final graph state is not a GraphState and could not be coerced: %r", final)
        except Exception as e:
            logger.exception("Error normalizing final graph state: %s", e)
        # persist assistant message (defensive: final may be GraphState or mapping-like)
        try:
            if isinstance(final, GraphState):
                summary_text = final.summary or ""
                prov = final.provenance or []
                ckpt_id = final.checkpoint_id if hasattr(final, "checkpoint_id") else None
            elif isinstance(final, dict):
                summary_text = final.get("summary") or ""
                prov = final.get("provenance") or []
                ckpt_id = final.get("checkpoint_id")
            else:
                # try mapping-like coercion
                try:
                    d = dict(final)
                    summary_text = d.get("summary") or ""
                    prov = d.get("provenance") or []
                    ckpt_id = d.get("checkpoint_id")
                except Exception:
                    summary_text = ""
                    prov = []
                    ckpt_id = None
        except Exception:
            summary_text = ""
            prov = []
            ckpt_id = None

        con.execute("INSERT INTO messages(message_id,session_id,role,content,agent) VALUES (?,?,?,?,?)",
                    (uuid.uuid4().hex, session_id, "assistant", summary_text, "summary"))
        con.commit(); con.close()

        # ensure provenance is JSON-serializable list
        try:
            if isinstance(prov, list):
                prov_out = prov
            else:
                prov_out = list(prov)
        except Exception:
            prov_out = []

        # If provenance indicates the summary agent ran but produced no text, provide
        # a defensive placeholder so the client doesn't render an empty assistant message.
        try:
            if (not summary_text or not str(summary_text).strip()) and any((p.get('agent')=='summary' for p in prov_out)):
                logger.warning("Session %s: summary provenance present but summary_text empty; inserting placeholder", session_id)
                summary_text = "(no summary generated)"
        except Exception:
            pass

        # Determine which agent produced the final output. Prefer the last provenance entry
        # if present, otherwise fall back to a generic 'assistant' label.
        try:
            if isinstance(prov_out, list) and len(prov_out) > 0:
                meta_agent = prov_out[-1].get('agent', 'assistant')
            else:
                meta_agent = 'assistant'
        except Exception:
            meta_agent = 'assistant'

        # Include the final assistant text in the meta payload so clients can show it
        # even when the graph did not stream token-level chunks.
        await ws.send_json({"type":"meta","data":{"agent": meta_agent, "text": summary_text, "provenance":prov_out,"checkpoint_id":ckpt_id}})
        await ws.send_json({"type":"done"})
    except Exception as e:
        await ws.send_json({"type":"error","data":str(e)})
        await ws.close()
