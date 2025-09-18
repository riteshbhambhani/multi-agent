from langchain.prompts import PromptTemplate
from .provenance import log_provenance
from models.model_loader import model_info

SUMMARY_PROMPT = PromptTemplate.from_template("""You are SummaryAgent. Merge BENEFIT and CLAIM results into 3 bullets max, then 'Next steps' list.
Benefit:
{benefit}
Claim:
{claim}
""")

class SummaryAgent:
    def __init__(self, llm):
        self.llm = llm
        info = model_info()
        self.model_name = info["model_name"]
        self.quant = info["quantization"]
    def run(self, state):
        b = state.benefit_result or ""
        c = state.claim_result or ""
        gen = self.llm.stream(SUMMARY_PROMPT.format(benefit=b, claim=c))
        out = ""
        for ch in gen: out += ch
        log_provenance(state.session_id,"summary",self.model_name,self.quant,[])
        return {"answer": out, "provenance":[{"agent":"summary","model":self.model_name,"quant":self.quant,"sources":[]}]}
