import os, logging
from langchain.prompts import PromptTemplate
from .retrieval import ClaimRetriever
from .provenance import log_provenance
from ..models.model_loader import model_info
import re
from .retrieval import find_claim_by_id

CLAIM_PROMPT = PromptTemplate.from_template("""You are ClaimAgent. Use ONLY the context to answer.
If the claim cannot be uniquely identified, ask ONE clarifying question (service date or provider).
Question: {question}
Context:
{context}
Return: Short reason (if any) + next steps.
""")

logger = logging.getLogger("backend.agents.claim")

class ClaimAgent:
    def __init__(self, llm):
        self.llm = llm
        self.ret = ClaimRetriever()
        info = model_info()
        self.model_name = info["model_name"]
        self.quant = info["quantization"]
        logger.info("ClaimAgent initialized with model=%s", self.model_name)
    def run(self, q, session_id, user_id):
        import time
        start_ts = time.time()
        logger.info("ClaimAgent.run start session=%s user=%s question=%s", session_id, user_id, (q or '')[:80])
        # Quick path: if the user provided an explicit claim id like 'claim_ad69f6a9', return exact fields.
        try:
            m = re.search(r"(claim_[0-9a-fA-F]+)", q)
            if m:
                cid = m.group(1)
                doc = find_claim_by_id(cid)
                if doc:
                    billed = doc.get('billed_amount')
                    allowed = doc.get('allowed_amount')
                    paid = doc.get('paid_amount')
                    status = doc.get('status')
                    prov = [{"file":"claims.json","doc_id": cid, "offsets": []}]
                    logger.info("ClaimAgent quick-lookup found claim=%s paid=%s status=%s", cid, paid, status)
                    answer = (
                        f"Claim {cid} details:\n"
                        f"- Billed amount: ${billed}\n"
                        f"- Allowed amount: ${allowed}\n"
                        f"- Paid amount: ${paid}\n"
                        f"- Status: {status}"
                    )
                    return {
                        "answer": answer,
                        "provenance": [{"agent":"claim","model":self.model_name,"quant":self.quant,"sources":prov}]
                    }
                else:
                    logger.info("ClaimAgent quick-lookup no match for %s", cid)
        except Exception:
            logger.exception("Error during claim quick-lookup")
        ctx, prov = self.ret.search(q, k=6)
        logger.info("ClaimAgent.run retrieved context for session=%s context_len=%d sources=%d", session_id, len(ctx or ""), len(prov or []))
        try:
            gen = self.llm.stream(CLAIM_PROMPT.format(question=q, context=ctx))
            out = ""
            chunks = 0
            first_token_logged = False
            for chunk in gen:
                # avoid logging every token; only log when streaming starts
                if not first_token_logged:
                    logger.info("ClaimAgent streaming started session=%s", session_id)
                    first_token_logged = True
                chunks += 1
                out += chunk
            duration = time.time() - start_ts
            log_provenance(session_id,"claim",self.model_name,self.quant,prov)
            logger.info("ClaimAgent.run completed session=%s chunks=%d answer_len=%d duration=%.2fs", session_id, chunks, len(out), duration)
            return {"answer": out, "provenance": [{"agent":"claim","model":self.model_name,"quant":self.quant,"sources":prov}]}
        except Exception as e:
            logger.exception("ClaimAgent.run error session=%s user=%s: %s", session_id, user_id, str(e))
            raise
