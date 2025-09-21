import time
from langchain.prompts import PromptTemplate
from .retrieval import ClaimRetriever
from backend.logging_setup import setup_logging


logger = setup_logging("ClaimAgent")


CLAIM_PROMPT = PromptTemplate.from_template("""You are ClaimAgent. Use ONLY the context to answer.
If the claim cannot be uniquely identified, ask ONE clarifying question (service date or provider).
Question: {question}
Context:
{context}
Return: Short reason (if any) + next steps.
""")


class ClaimAgent:
    def __init__(self, llm):
        self.llm = llm
        self.ret = ClaimRetriever()
        self.model_name = getattr(getattr(llm, '__class__', object), '__name__', 'LLM')
        self.quant = None
        logger.info("ClaimAgent initialized")


    def run(self, q: str, session_id: str, user_id: str):
        start_ts = time.time()
        
        logger.info("ClaimAgent.run start session=%s user=%s", session_id, user_id)
        ctx, prov = self.ret.search(q, k=20, final_k=5)
        logger.info("ClaimAgent.run with cintext question=%s context=%s", q, ctx)
        prompt = CLAIM_PROMPT.format(question=q, context=ctx)
        out = "".join(self.llm.stream(prompt))
        logger.info("ClaimAgent.run completed in %.2fs", time.time()-start_ts)
        return {"answer": out, "provenance":[{"agent":"claim","model":self.model_name,"quant":self.quant,"sources":prov}]}