import argparse, json, sqlite3, os, pathlib, uuid, logging
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger("backend.scripts.ingest")

parser = argparse.ArgumentParser()
parser.add_argument("--benefits", required=True)
parser.add_argument("--claims", required=True)
args = parser.parse_args()
logger.info("Ingest started benefits=%s claims=%s", args.benefits, args.claims)

DB_PATH = os.getenv("DB_PATH","backend/db/app.db")
EMB_NAME = os.getenv("EMBEDDINGS_MODEL","BAAI/bge-small-en-v1.5")
INDEX_DIR = pathlib.Path("backend/db/index"); INDEX_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = INDEX_DIR / "main.faiss"

def load_json(p):
    return json.loads(pathlib.Path(p).read_text())

benefits = load_json(args.benefits)
claims = load_json(args.claims)
logger.info("Loaded %d benefits and %d claims from disk", len(benefits), len(claims))

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

def insert_docs(items, dtype, source):
    for it in items:
        doc_id = it.get("claim_id") or it.get("benefit_id") or uuid.uuid4().hex
        cur.execute("INSERT OR REPLACE INTO documents(doc_id,source_file,doc_type,content) VALUES (?,?,?,?)",
            (doc_id, source, dtype, json.dumps(it)))
    con.commit()

insert_docs(benefits, "benefit", "benefits.json")
insert_docs(claims, "claim", "claims.json")

# Build vector index
texts, metas = [], []
for row in cur.execute("SELECT doc_id, doc_type, source_file, content FROM documents"):
    doc_id, dt, src, content = row
    texts.append(content)
    metas.append({"id":doc_id, "doc_type":dt, "source":src})

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
docs = []
for t,m in zip(texts, metas):
    for chunk in splitter.split_text(t):
        docs.append(Document(page_content=chunk, metadata=m))

model = SentenceTransformer(EMB_NAME, device="cpu")
class STEmb(Embeddings):
    def embed_documents(self, texts): return model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text): return model.encode([text], normalize_embeddings=True)[0].tolist()

vs = FAISS.from_documents(docs, STEmb())
vs.save_local(str(INDEX_PATH))

cur.execute("INSERT OR REPLACE INTO vector_index(idx_name, location, dim) VALUES (?,?,?)",
            ("main", str(INDEX_PATH), model.get_sentence_embedding_dimension()))
con.commit(); con.close()
logger.info("Ingest complete. Index saved at %s", INDEX_PATH)
print("Ingested", len(benefits), "benefits and", len(claims), "claims. Index at", INDEX_PATH)
