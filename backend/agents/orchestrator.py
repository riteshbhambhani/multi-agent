import re, json, os
from pydantic import BaseModel
from typing import Literal, Optional, List
from langgraph.graph import StateGraph, END
from tenacity import retry, wait_exponential, stop_after_attempt
import logging
from math import acos
from math import sqrt

logger = logging.getLogger("backend.orchestrator")

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
    route_confidence: Optional[float] = None

benefit_terms = r"(benefit|coverage|copay|coinsurance|deductible|eligibility|in[- ]network|out[- ]of[- ]pocket)"
claim_terms   = r"(claim|eob|denied|allowed|paid|adjusted|appeal|authorization|prior auth)"

def _route(q:str)->str:
    has_b = re.search(benefit_terms, q, re.I) is not None
    has_c = re.search(claim_terms, q, re.I) is not None
    if has_b and has_c: return "both"
    if has_b: return "benefit"
    if has_c: return "claim"
    return "clarify"


class SemanticRouter:
    """Lightweight semantic router using sentence-transformers embeddings.
    It loads a small embeddings model and compares cosine similarity to prototype
    sentences for 'benefit' and 'claim'. If the model cannot be loaded it will
    raise and the caller should fall back to the regex router.
    """
    def __init__(self, model_name: str = None):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            raise
        self.model_name = model_name or os.getenv('EMBEDDINGS_MODEL','sentence-transformers/all-MiniLM-L6-v2')
        self.enc = SentenceTransformer(self.model_name)
        # prototype sentences representing each class
        self.protos = {
            'benefit': [
                'Is this about coverage, copay, deductible, in-network or out-of-pocket?',
                'What is covered under my plan?',
            ],
            'claim': [
                'Is this about a specific claim, EOB, denied or paid amount?',
                'Why was my claim denied or what was paid?',
            ],
            'both': [
                'I have a question that involves both coverage and a specific claim',
            ]
        }
        # embed prototypes
        self.proto_emb = {k: [self._norm(e) for e in self.enc.encode(v, convert_to_numpy=True)] for k,v in self.protos.items()}

    def _norm(self, vec):
        # return L2-normalized vector
        import numpy as _np
        arr = _np.array(vec, dtype=float)
        n = _np.linalg.norm(arr)
        return arr / (n+1e-12)

    def classify(self, text: str) -> tuple:
        import numpy as _np
        qv = self._norm(self.enc.encode([text], convert_to_numpy=True)[0])
        scores = {}
        for k, vlist in self.proto_emb.items():
            # cosine similarity averaged
            sims = [_np.dot(qv, pv) for pv in vlist]
            scores[k] = float(_np.mean(sims))
        # choose best
        best = max(scores.items(), key=lambda x: x[1])
        best_label, best_score = best[0], best[1]
        # thresholding to detect clarify vs strong match
        if best_score < float(os.getenv('ROUTER_CLARIFY_THRESHOLD','0.30')):
            return ('clarify', float(best_score))
        # if both benefit and claim are both reasonably high, return 'both'
        if scores.get('benefit',0) > float(os.getenv('ROUTER_BOTH_THRESHOLD','0.45')) and scores.get('claim',0) > float(os.getenv('ROUTER_BOTH_THRESHOLD','0.45')):
            # return the avg of the two scores as confidence
            return ('both', float((scores.get('benefit')+scores.get('claim'))/2.0))
        return (best_label, float(best_score))

def router_node(state: GraphState):
    # Prefer semantic routing when possible, otherwise fall back to regex-based routing.
    try:
        # cache a router on the module to avoid repeated loads
        if not hasattr(router_node, '_sr'):
            router_node._sr = SemanticRouter()
        sr = router_node._sr
        lab, conf = sr.classify(state.question)
        state.route = lab
        state.route_confidence = conf
    except Exception:
        # model not available or failed — fallback to regex
        state.route = _route(state.question)
        state.route_confidence = 1.0
    logger.info("Router determined route=%s for session=%s", state.route, getattr(state, "session_id", None))
    return state

@retry(wait=wait_exponential(multiplier=0.25, min=0.25, max=1), stop=stop_after_attempt(2))
def benefit_node(state, benefit_agent):
    res = benefit_agent.run(state.question, state.session_id, state.user_id)
    state.benefit_result = res["answer"]; state.provenance += res["provenance"]
    logger.info("Benefit node completed for session=%s", state.session_id)
    return state

@retry(wait=wait_exponential(multiplier=0.25, min=0.25, max=1), stop=stop_after_attempt(2))
def claim_node(state, claim_agent):
    res = claim_agent.run(state.question, state.session_id, state.user_id)
    state.claim_result = res["answer"]; state.provenance += res["provenance"]
    logger.info("Claim node completed for session=%s", state.session_id)
    return state

def summary_node(state, summary_agent):
    res = summary_agent.run(state)
    state.summary = res["answer"]; state.provenance += res["provenance"]
    logger.info("Summary node completed for session=%s", state.session_id)
    return state

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
    g.add_node("summary_node", lambda s: summary_node(s, summary_agent))
    g.add_node("clarify", lambda s: clarification_node(s, ckpt_store))
    g.set_entry_point("router")
    g.add_conditional_edges("router", lambda s: s.route, {
        "benefit": "benefit",
        "claim": "claim",
        # map 'both' to a single start node (sequential flow: benefit -> claim)
        "both": "benefit",
        "clarify": "clarify",
        "unknown": "clarify"
    })
    # chain benefit -> claim -> summary_node for 'both' behaviour
    g.add_edge("benefit", "claim")
    g.add_edge("claim", "summary_node")
    g.add_edge("clarify", END)
    g.add_edge("summary_node", END)
    return g.compile()
