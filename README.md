# Healthcare Benefits & Claims Multi-Agent App (LangGraph + LangChain + MCP + RAGAS)

**Stack**: FastAPI (Python 3.11), LangGraph, LangChain, llama.cpp (HF GGUF), FAISS, SentenceTransformers, SQLite, MCP (stubs), React (Vite + TS).  
**Hardware target**: Apple MacBook Air M4 24GB RAM — all inference local by default.

## Quickstart
```bash
# 1) Python env
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

# 2) Configure
cp backend/.env.example backend/.env
# Edit backend/.env -> set MODEL_PATH to local GGUF (e.g. ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf)

# 3) Initialize DB and ingest sample data (optional: use the ZIP I shared earlier)
python backend/scripts/init_db.py
python backend/scripts/ingest.py --benefits backend/data/benefits.json --claims backend/data/claims.json

# 4) Run backend
uvicorn backend.main:app --reload

# 5) Frontend
(cd frontend && npm i && npm run dev)
```

## Notes
- Local-only by default. To allow HF API fallback, set `ALLOW_REMOTE_LLMS=true` in `.env` (not recommended for PHI).
- Tests: `pytest -q` from backend dir.
