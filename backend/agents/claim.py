import os
from langchain.prompts import PromptTemplate
from .retrieval import ClaimRetriever
from .provenance import log_provenance
from models.model_loader import model_info

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
        info = model_info()
        self.model_name = info["model_name"]
        self.quant = info["quantization"]
    def run(self, q, session_id, user_id):
        ctx, prov = self.ret.search(q, k=6)
        gen = self.llm.stream(CLAIM_PROMPT.format(question=q, context=ctx))
        out = ""
        for chunk in gen:
            out += chunk
        log_provenance(session_id,"claim",self.model_name,self.quant,prov)
        return {"answer": out, "provenance": [{"agent":"claim","model":self.model_name,"quant":self.quant,"sources":prov}]}
