import re, json
from pydantic import BaseModel
from typing import Literal, Optional, List
from langgraph.graph import StateGraph, END
from tenacity import retry, wait_exponential, stop_after_attempt

class GraphState(BaseModel):
    session_id: str
    user_id: str
    question: str
    route: Literal["benefit","claim","both","clarify","unknown"] = "unknown"
    benefit_result: Optional[str] = None
    claim_result: Optional[str] = None
    summary: Optional[str] = None
    provenance: List[dict] = []
    checkpoint_id: Optional[str] = None

benefit_terms = r"(benefit|coverage|copay|coinsurance|deductible|eligibility|in[- ]network|out[- ]of[- ]pocket)"
claim_terms   = r"(claim|eob|denied|allowed|paid|adjusted|appeal|authorization|prior auth)"

def _route(q:str)->str:
    has_b = re.search(benefit_terms, q, re.I) is not None
    has_c = re.search(claim_terms, q, re.I) is not None
    if has_b and has_c: return "both"
    if has_b: return "benefit"
    if has_c: return "claim"
    return "clarify"

def router_node(state: GraphState):
    state.route = _route(state.question); return state

@retry(wait=wait_exponential(multiplier=0.25, min=0.25, max=1), stop=stop_after_attempt(2))
def benefit_node(state, benefit_agent):
    res = benefit_agent.run(state.question, state.session_id, state.user_id)
    state.benefit_result = res["answer"]; state.provenance += res["provenance"]; return state

@retry(wait=wait_exponential(multiplier=0.25, min=0.25, max=1), stop=stop_after_attempt(2))
def claim_node(state, claim_agent):
    res = claim_agent.run(state.question, state.session_id, state.user_id)
    state.claim_result = res["answer"]; state.provenance += res["provenance"]; return state

def summary_node(state, summary_agent):
    res = summary_agent.run(state)
    state.summary = res["answer"]; state.provenance += res["provenance"]; return state

def clarification_node(state, ckpt_store):
    ckpt = ckpt_store.create(
        user_id=state.user_id,
        session_id=state.session_id,
        pending_agent="orchestrator",
        pending_question="Is your question about benefits, claims, or both?",
        context_snapshot=state.model_dump_json()
    )
    state.checkpoint_id = ckpt["checkpoint_id"]; return state

def build_graph(benefit_agent, claim_agent, summary_agent, ckpt_store):
    g = StateGraph(GraphState)
    g.add_node("router", router_node)
    g.add_node("benefit", lambda s: benefit_node(s, benefit_agent))
    g.add_node("claim",   lambda s: claim_node(s, claim_agent))
    g.add_node("summary", lambda s: summary_node(s, summary_agent))
    g.add_node("clarify", lambda s: clarification_node(s, ckpt_store))
    g.set_entry_point("router")
    g.add_conditional_edges("router", lambda s: s.route, {
        "benefit": "benefit",
        "claim": "claim",
        "both": ["benefit","claim"],
        "clarify": "clarify",
        "unknown": "clarify"
    })
    g.add_edge("benefit", "summary")
    g.add_edge("claim", "summary")
    g.add_edge("clarify", END)
    g.add_edge("summary", END)
    return g.compile(parallel=True)
