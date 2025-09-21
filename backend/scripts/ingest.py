import os, json, pathlib, chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_PATH = pathlib.Path(os.getenv("CHROMA_PATH", "backend/db/chroma")).resolve()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

client = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(allow_reset=True))
embed = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

def embed_text(text): return embed.encode([text], normalize_embeddings=True).tolist()[0]

def load_and_ingest(collection_name, file_path, doc_type):
    coll = client.get_or_create_collection(collection_name)
    with open(file_path, "r") as f: data = json.load(f)

    for rec in data:
        if doc_type == "claims":
            # Flatten claim lines if present
            claim_text = f"Claim ID: {rec['claim_id']}, Member: {rec['member_id']}, Provider: {rec['provider']}, Status: {rec['status']}, Billed: {rec['billed_amount']}, Allowed: {rec['allowed_amount']}, Paid: {rec['paid_amount']}"
            if rec.get("denial_reason"):
                claim_text += f", Denial Reason: {rec['denial_reason']}"
            if rec.get("claim_lines"):
                for line in rec["claim_lines"]:
                    claim_text += f"\n  Line: {line['procedure_code']} billed {line['billed_amount']} allowed {line['allowed_amount']} paid {line['paid_amount']}"

            meta = {
                "member_id": rec["member_id"],
                "status": rec["status"],
                "out_of_network": bool(rec.get("out_of_network", False)),
                "denial_reason": rec.get("denial_reason") or "",  # convert None -> ""
                "icd": rec.get("icd") or ""                      # convert None -> ""
            }
            coll.add(
                ids=[rec["claim_id"]],
                documents=[claim_text],
                metadatas=[meta],
                embeddings=[embed_text(claim_text)]
            )

        elif doc_type == "benefits":
            benefit_text = f"Member {rec['member_id']} has plan {rec['plan_name']} effective {rec['effective_date']}, OOP max {rec['out_of_pocket_max']}, Deductible remaining {rec['deductible_remaining']}."
            meta = {
                "member_id": rec["member_id"],
                "plan_id": rec["plan_id"],
                "in_network": rec["in_network"]
            }
            coll.add(
                ids=[rec["benefit_id"]],
                documents=[benefit_text],
                metadatas=[meta],
                embeddings=[embed_text(benefit_text)]
            )

if __name__ == "__main__":
    load_and_ingest("claims", "backend/data/claims_synthetic.json", "claims")
    load_and_ingest("benefits", "backend/data/benefits.json", "benefits")
    print("Ingestion complete.")