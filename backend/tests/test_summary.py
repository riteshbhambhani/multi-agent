from ..agents.summary import SummaryAgent
from ..agents.orchestrator import GraphState
from ..models.model_loader import load_llm
import os, pytest

@pytest.mark.skipif(not os.getenv("MODEL_PATH"), reason="MODEL_PATH not set")
def test_summary_runs():
    llm = load_llm()
    sa = SummaryAgent(llm)
    st = GraphState(session_id="s", user_id="u", question="q", benefit_result="Benefit ok", claim_result="Denied")
    out = sa.run(st)
    assert "Next" in out["answer"] or out["answer"]
