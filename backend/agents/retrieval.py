import os, pathlib, re
from typing import List, Tuple, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from backend.logging_setup import setup_logging


logger = setup_logging("retrieval")


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHROMA_PATH = pathlib.Path(os.getenv("CHROMA_PATH", "backend/db/chroma")).resolve()
TOP_K = int(os.getenv("RETRIEVE_K", "20"))
FINAL_K = int(os.getenv("FINAL_K", "5"))

MEMBER_ID_REGEX = re.compile(r"\bM\d{6}\b")  # e.g., M770487


class ChromaRetriever:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH), settings=Settings(allow_reset=False)
        )
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embed = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        self.reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
        logger.info("Retriever ready: collection=%s", collection_name)

    def _query(
        self, query: str, k: int = TOP_K, where: Optional[Dict] = None
    ) -> List[Tuple[str, str, Dict]]:
        qv = self.embed.encode([query], normalize_embeddings=True).tolist()[0]
        res = self.collection.query(
            query_embeddings=[qv],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0] if "ids" in res else [m.get("id", "unknown") for m in metas]
        out = [(i, d, m) for i, d, m in zip(ids, docs, metas)]
        logger.debug(
            "Retrieved %d candidates from %s with filter=%s",
            len(out), self.collection_name, where,
        )
        return out

    def search(
        self, query: str, k: int = TOP_K, final_k: int = FINAL_K
    ) -> Tuple[str, List[Dict]]:
        cands = self._query(query, k=k)
        if not cands:
            return "", []
        pairs = [(query, txt) for _, txt, _ in cands]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
        top = ranked[:final_k]
        context = "\n\n".join(item[0][1] for item in top)
        prov = [
            {"file": self.collection_name, "doc_id": item[0][0], "offsets": []}
            for item in top
        ]
        logger.info(
            "Reranked %d→%d for %s", len(cands), len(top), self.collection_name
        )
        return context, prov


class BenefitRetriever(ChromaRetriever):
    def __init__(self):
        super().__init__("benefits")


class ClaimRetriever(ChromaRetriever):
    def __init__(self):
        super().__init__("claims")

    def search(
        self, query: str, k: int = TOP_K, final_k: int = FINAL_K
    ) -> Tuple[str, List[Dict]]:
        # Try to extract member_id
        member_match = MEMBER_ID_REGEX.search(query)
        where = {"member_id": member_match.group()} if member_match else None
        if where:
            logger.info("Applying hybrid filter: restricting to member_id=%s", where["member_id"])
        else:
            logger.info("No member_id found in query. Running pure semantic search.")

        # Step 1: Candidate retrieval
        cands = self._query(query, k=k, where=where)
        if not cands:
            return "", []

        # Step 2: Reranking
        pairs = [(query, txt) for _, txt, _ in cands]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
        top = ranked[:final_k]

        # Step 3: Build context + provenance
        context = "\n\n".join(item[0][1] for item in top)
        prov = [
            {
                "file": self.collection_name,
                "doc_id": item[0][0],
                "member_id": item[0][2].get("member_id"),
                "offsets": [],
            }
            for item in top
        ]
        logger.info(
            "Reranked %d→%d for %s (filter=%s)",
            len(cands), len(top), self.collection_name, where,
        )
        return context, prov
