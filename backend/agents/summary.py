import logging
from langchain.prompts import PromptTemplate
from .provenance import log_provenance
from ..models.model_loader import model_info

logger = logging.getLogger("backend.agents.summary")

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
        logger.info("SummaryAgent initialized with model=%s", self.model_name)
    def run(self, state):
        import time
        b = state.benefit_result or ""
        c = state.claim_result or ""
        start_ts = time.time()
        logger.info("SummaryAgent.run start session=%s", state.session_id)
        try:
            gen = self.llm.stream(SUMMARY_PROMPT.format(benefit=b, claim=c))
            out = ""
            chunks = 0
            started = False
            for ch in gen:
                if not started:
                    logger.info("SummaryAgent streaming started session=%s", state.session_id)
                    started = True
                chunks += 1
                out += ch
            duration = time.time() - start_ts
            duration = time.time() - start_ts
            # Defensive fallback: if the model produced no text, return an explicit placeholder
            if not out or not out.strip():
                logger.warning("SummaryAgent produced empty output for session=%s; returning placeholder", state.session_id)
                out = "(no summary generated)"
            log_provenance(state.session_id,"summary",self.model_name,self.quant,[])
            logger.info("SummaryAgent.run completed session=%s chunks=%d answer_len=%d duration=%.2fs", state.session_id, chunks, len(out), duration)
            return {"answer": out, "provenance":[{"agent":"summary","model":self.model_name,"quant":self.quant,"sources":[]}]} 
        except Exception as e:
            logger.exception("SummaryAgent.run error session=%s: %s", state.session_id, str(e))
            raise
