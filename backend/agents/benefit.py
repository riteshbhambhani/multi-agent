import os
from langchain.prompts import PromptTemplate
from .retrieval import BenefitRetriever
from .provenance import log_provenance
from models.model_loader import model_info

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
    def run(self, q, session_id, user_id):
        ctx, prov = self.ret.search(q, k=6)
        gen = self.llm.stream(BENEFIT_PROMPT.format(question=q, context=ctx))
        out = ""
        for chunk in gen:
            out += chunk
        log_provenance(session_id,"benefit",self.model_name,self.quant,prov)
        return {"answer": out, "provenance": [{"agent":"benefit","model":self.model_name,"quant":self.quant,"sources":prov}]}
