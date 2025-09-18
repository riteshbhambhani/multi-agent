import os, logging
from langchain.prompts import PromptTemplate
from .retrieval import BenefitRetriever, find_benefit_by_id
import re
from .provenance import log_provenance
from ..models.model_loader import model_info

logger = logging.getLogger("backend.agents.benefit")

BENEFIT_PROMPT = PromptTemplate.from_template("""You are BenefitAgent. Use ONLY provided context to answer.
Question: {question}
Context:
{context}
Respond with up to 4 bullets, factual policy language. If missing data, ask ONE specific question.
""")

class BenefitAgent:
    def __init__(self, llm):
        self.llm = llm
        self.ret = BenefitRetriever()
        info = model_info()
        self.model_name = info["model_name"]
        self.quant = info["quantization"]
        logger.info("BenefitAgent initialized with model=%s", self.model_name)
    def run(self, q, session_id, user_id):
        import time
        start_ts = time.time()
        logger.info("BenefitAgent.run start session=%s user=%s question=%s", session_id, user_id, (q or '')[:80])
        # Quick path: if the user provided an explicit benefit id like 'benefit_79a87fe2', return structured fields.
        try:
            m = re.search(r"(benefit_[0-9a-fA-F]+)", q)
            if m:
                bid = m.group(1)
                doc = find_benefit_by_id(bid)
                if doc:
                    plan = doc.get('plan_name')
                    oop = doc.get('out_of_pocket_max')
                    ded_rem = doc.get('deductible_remaining')
                    coverages = doc.get('coverages') or []
                    prov = [{"file":"benefits.json","doc_id": bid, "offsets": []}]
                    coverage_lines = "\n".join([f"  - {c.get('category')}: copay ${c.get('copay')}, coinsurance {c.get('coinsurance')}, deductible ${c.get('deductible')}" for c in coverages])
                    answer = (
                        f"Benefit {bid} details:\n"
                        f"- Plan: {plan}\n"
                        f"- Out-of-pocket max: ${oop}\n"
                        f"- Deductible remaining: ${ded_rem}\n"
                        f"- Coverages:\n{coverage_lines}"
                    )
                    logger.info("BenefitAgent quick-lookup found benefit=%s plan=%s", bid, plan)
                    return {"answer": answer, "provenance": [{"agent":"benefit","model":self.model_name,"quant":self.quant,"sources":prov}]}
        except Exception:
            logger.exception("Error during benefit quick-lookup")
        ctx, prov = self.ret.search(q, k=6)
        logger.info("BenefitAgent.run retrieved context for session=%s context_len=%d sources=%d", session_id, len(ctx or ""), len(prov or []))
        try:
            gen = self.llm.stream(BENEFIT_PROMPT.format(question=q, context=ctx))
            out = ""
            chunks = 0
            first_token = False
            for chunk in gen:
                if not first_token:
                    logger.info("BenefitAgent streaming started session=%s", session_id)
                    first_token = True
                chunks += 1
                out += chunk
            duration = time.time() - start_ts
            log_provenance(session_id,"benefit",self.model_name,self.quant,prov)
            logger.info("BenefitAgent.run completed session=%s chunks=%d answer_len=%d duration=%.2fs", session_id, chunks, len(out), duration)
            return {"answer": out, "provenance": [{"agent":"benefit","model":self.model_name,"quant":self.quant,"sources":prov}]}
        except Exception as e:
            logger.exception("BenefitAgent.run error session=%s user=%s: %s", session_id, user_id, str(e))
            raise
