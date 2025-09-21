import time
from langchain.prompts import PromptTemplate
from .retrieval import BenefitRetriever
from backend.logging_setup import setup_logging


logger = setup_logging("BenefitAgent")


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
        self.model_name = getattr(getattr(llm, '__class__', object), '__name__', 'LLM')
        self.quant = None
        logger.info("BenefitAgent initialized")


    def run(self, q: str, session_id: str, user_id: str):
        start_ts = time.time()
        logger.info("BenefitAgent.run start session=%s user=%s", session_id, user_id)
        ctx, prov = self.ret.search(q, k=20, final_k=5)
        prompt = BENEFIT_PROMPT.format(question=q, context=ctx)
        out = "".join(self.llm.stream(prompt))
        logger.info("BenefitAgent.run completed in %.2fs", time.time()-start_ts)
        return {"answer": out, "provenance":[{"agent":"benefit","model":self.model_name,"quant":self.quant,"sources":prov}]}