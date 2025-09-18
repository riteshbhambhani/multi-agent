import json, sqlite3, os, pathlib, logging
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

DB_PATH = os.getenv("DB_PATH","backend/db/app.db")
logger = logging.getLogger("backend.retrieval")

class STEmb(Embeddings):
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        logger.info("Initializing SentenceTransformer embeddings model=%s", name)
        self.m = SentenceTransformer(name, device="cpu")
    def embed_documents(self, texts): return self.m.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text): return self.m.encode([text], normalize_embeddings=True)[0].tolist()

def load_faiss():
    logger.info("Loading FAISS index from DB=%s", DB_PATH)
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT location FROM vector_index WHERE idx_name='main'").fetchone()
    con.close()
    if not row:
        logger.error("No vector index found in DB; instruct to run scripts/ingest.py")
        raise RuntimeError("Vector index not found. Run scripts/ingest.py")
    index_path = row[0]
    emb_name = os.getenv("EMBEDDINGS_MODEL","BAAI/bge-small-en-v1.5")
    logger.info("FAISS index path=%s embeddings_model=%s", index_path, emb_name)
    return FAISS.load_local(index_path, STEmb(emb_name), allow_dangerous_deserialization=True)

class BenefitRetriever:
    def __init__(self):
        logger.info("Initializing BenefitRetriever")
        self.index = load_faiss()
        logger.info("BenefitRetriever initialized")
    def search(self, query, k=6):
        logger.info("BenefitRetriever.search query=%s k=%d", query[:80], k)
        docs = self.index.similarity_search(query, k=k)
        ctx = "\n\n".join(d.page_content for d in docs)
        prov = [{"file":d.metadata.get("source"),"doc_id":d.metadata.get("id"),"offsets":[]} for d in docs]
        logger.info("BenefitRetriever.search returned %d docs", len(docs))
        return ctx, prov

class ClaimRetriever(BenefitRetriever):
    pass

def load_claims_data():
    p = os.path.join(os.path.dirname(__file__), '..', 'data', 'claims.json')
    p = os.path.normpath(p)
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load claims.json at %s", p)
        return []

def find_claim_by_id(claim_id: str):
    claims = load_claims_data()
    for c in claims:
        # normalized key may be 'claim_id' or 'id'
        cid = c.get('claim_id') or c.get('id')
        if cid == claim_id:
            return c
    return None


def load_benefits_data():
    p = os.path.join(os.path.dirname(__file__), '..', 'data', 'benefits.json')
    p = os.path.normpath(p)
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load benefits.json at %s", p)
        return []


def find_benefit_by_id(benefit_id: str):
    benefits = load_benefits_data()
    for b in benefits:
        bid = b.get('benefit_id') or b.get('id')
        if bid == benefit_id:
            return b
    return None
