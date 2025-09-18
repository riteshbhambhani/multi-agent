Multi-Agent demo — local development guide

This repository contains a small multi-agent demo (FastAPI backend + Vite frontend) that shows how simple agents (benefit, claim, summary) coordinate to answer user questions. This README focuses on getting a local development environment running for a novice.

Prerequisites
- Git
- Python 3.11 (recommended). Using Python 3.12+ may cause some packages (tokenizers) to build from source and fail.
- Node.js 18+ and npm (for frontend)
- (Optional) A Hugging Face token if you want to use remote HF router/inference: sign in at https://huggingface.co/settings/tokens and create a token.

Quick start (macOS / zsh)
1. Clone the repo

   git clone <your-repo-url>
   cd multi-agent

2. Create a Python 3.11 virtual environment and activate it

   # macOS / zsh
   python3.11 -m venv ./.venv
   source ./.venv/bin/activate

   If you don't have python3.11, install it (homebrew or pyenv). This project targets 3.11 to avoid Rust/PyO3 build issues with tokenizers.

3. Install backend dependencies

   pip install --upgrade pip
   pip install -r backend/requirements.txt

4. (Optional) Provide a Hugging Face token

   - Create a file `backend/.env` with:
       HF_TOKEN=hf_...
       HF_MODE=router         # or 'inference_api' or 'transformers'
       HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct

   - If you want to use local transformers (no HF_TOKEN), set HF_MODE=transformers and ensure you have enough disk/RAM for the chosen model.

5. Start the backend

   # from project root
   ./.venv/bin/python -m uvicorn backend.main:app --reload

   The backend opens on http://127.0.0.1:8000 by default.

6. Start the frontend

   cd frontend
   npm install
   npm run dev

   The frontend dev server runs on http://127.0.0.1:5173 and will call the backend at http://127.0.0.1:8000 by default.

7. Try the websocket integration test (quick smoke)

   # from project root (backend server must be running)
   ./.venv/bin/python scripts/ws_integration_test.py

Troubleshooting
- Tokenizers / Rust build errors: Ensure Python 3.11 is used. If you see errors about PyO3 or tokenizers compiling for Python 3.13+, switch to 3.11.
- FAISS install errors: On macOS, prebuilt wheels may not exist for some combinations. The repository pins a working FAISS wheel in `backend/requirements.txt`. If install still fails, consider using a Docker container or a linux VM, or use CPU-only compatible wheels.
- Hugging Face 404 or permission errors: Some models (e.g., Qwen) require accepting model terms on huggingface.co or using a router key. If you see HfHubHTTPError 404/403, verify `HF_MODEL_ID` and `HF_TOKEN` and accept model terms on the model's page.

Developer notes
- Agent quick lookups: asking for explicit IDs like `claim_ad69f6a9` or `benefit_79a87fe2` triggers deterministic lookups from `backend/data/*.json` and returns structured fields with provenance.
- Semantic router: the orchestrator uses a small sentence-transformers model to classify questions into `benefit`, `claim`, `both`, or `clarify`. If that model cannot be loaded, the code falls back to a regex-based router.
- Logs: backend logs are written to `backend/logs/app.log` and stream to the console.

If you want, I can:
- Add a `Makefile` or `scripts/setup_env.sh` to automate venv creation and installs.
- Add a dedicated API for direct claim/benefit lookups (e.g., `/api/claims/{id}`), and unit tests.

Enjoy — ask here if anything fails and paste console errors so I can help debug.
# Healthcare Benefits & Claims Multi-Agent App (LangGraph + LangChain + MCP + RAGAS)

**Stack**: FastAPI (Python 3.11), LangGraph, LangChain, llama.cpp (HF GGUF), FAISS, SentenceTransformers, SQLite, MCP (stubs), React (Vite + TS).  
**Hardware target**: Apple MacBook Air M4 24GB RAM — all inference local by default.

## Quickstart
```bash
# 1) Python env
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

# macOS / Apple Silicon note:
# If pip fails to find a matching `faiss-cpu` wheel (common on macOS / M1/M2/M4),
# a reliable fallback is to install FAISS from conda-forge in a separate env or
# using `mamba`:
#
# conda create -n magent python=3.11 -y
# conda activate magent
# conda install -c conda-forge faiss-cpu -y
# pip install -r backend/requirements.txt  # then install remaining deps

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
