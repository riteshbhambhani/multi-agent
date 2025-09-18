import json, sqlite3, os, pathlib
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

DB_PATH = os.getenv("DB_PATH","backend/db/app.db")

class STEmb(Embeddings):
    def __init__(self, name): 
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name, device="cpu")
    def embed_documents(self, texts): return self.m.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text): return self.m.encode([text], normalize_embeddings=True)[0].tolist()

def load_faiss():
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT location FROM vector_index WHERE idx_name='main'").fetchone()
    con.close()
    if not row: raise RuntimeError("Vector index not found. Run scripts/ingest.py")
    index_path = row[0]
    emb_name = os.getenv("EMBEDDINGS_MODEL","BAAI/bge-small-en-v1.5")
    return FAISS.load_local(index_path, STEmb(emb_name), allow_dangerous_deserialization=True)

class BenefitRetriever:
    def __init__(self):
        self.index = load_faiss()
    def search(self, query, k=6):
        docs = self.index.similarity_search(query, k=k)
        ctx = "\n\n".join(d.page_content for d in docs)
        prov = [{"file":d.metadata["source"],"doc_id":d.metadata.get("id"),"offsets":[]} for d in docs]
        return ctx, prov

class ClaimRetriever(BenefitRetriever):
    pass
