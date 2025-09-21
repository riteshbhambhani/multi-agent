from __future__ import annotations

import logging
import re
import uuid
import json
import os
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from backend.logging_setup import setup_logging
from backend.agents import ckpt_store

# Reduce noisy HF tokenizers warning in forked workers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = setup_logging("orchestrator")


class GraphState(BaseModel):
    session_id: str
    user_id: str
    question: str
    route: Literal["benefit", "claim", "both", "clarify", "unknown"] = "unknown"
    original_route: Optional[str] = None
    needs_claim: bool = False

    benefit_result: Optional[str] = None
    claim_result: Optional[str] = None
    summary: Optional[str] = None
    provenance: List[dict] = Field(default_factory=list)
    checkpoint_id: Optional[str] = None


# ---------------------------
# Helpers
# ---------------------------

def contains_kw(q: str, kws: list[str]) -> bool:
    """Check if question contains any of the keywords."""
    return any(re.search(rf"\b{re.escape(w)}\b", q) for w in kws)


# ---------------------------
# Node functions
# ---------------------------

def router_node(state: GraphState) -> GraphState:
    q = state.question.lower()

    benefit_kw = ["benefit", "benefits", "coverage", "copay", "coinsurance", "deductible", "plan"]
    claim_kw = ["claim", "claims", "eob", "paid", "allowed", "denied", "provider", "service date"]

    b = contains_kw(q, benefit_kw)
    c = contains_kw(q, claim_kw)

    if b and c:
        decided = "both"
    elif b:
        decided = "benefit"
    elif c:
        decided = "claim"
    else:
        decided = "clarify"

    state.route = decided
    state.original_route = decided
    state.needs_claim = decided == "both"

    logger.info("Router decided route=%s for q='%s'", state.route, state.question)
    return state


def save_checkpoint(state: GraphState, agent: str) -> GraphState:
    snapshot = json.dumps(state.dict())
    ckpt = ckpt_store.create(
        user_id=state.user_id,
        session_id=state.session_id,
        pending_agent=agent,
        pending_question=state.question,
        context_snapshot=snapshot,
    )
    state.checkpoint_id = ckpt["checkpoint_id"]
    logger.debug("Checkpoint saved: %s for agent=%s", state.checkpoint_id, agent)
    return state


def claim_node(state: GraphState, agent) -> GraphState:
    logger.info(">>> Entered claim_node with question=%s", state.question)
    res = agent.run(state.question, state.session_id, state.user_id)
    state.claim_result = res["answer"]
    state.provenance += res.get("provenance", [])
    save_checkpoint(state, "claim")
    logger.info(
        "After claim_node: route=%s original_route=%s needs_claim=%s",
        state.route, state.original_route, state.needs_claim
    )
    return state


def summary_node(state: GraphState) -> GraphState:
    logger.info(
        ">>> Entered summary_node with benefit_result=%s and claim_result=%s",
        state.benefit_result, state.claim_result
    )
    summ = (
        f"Benefit:\n{state.benefit_result or '(none)'}\n\n"
        f"Claim:\n{state.claim_result or '(none)'}"
    )
    state.summary = summ
    save_checkpoint(state, "summary")
    logger.info("After summary_node: checkpoint_id=%s", state.checkpoint_id)
    return state


def noop_node(state: GraphState) -> GraphState:
    """No-op node to satisfy LangGraph's 'no dead-end' validation."""
    logger.debug(">>> Entered noop_node (pass-through)")
    return state


# ---------------------------
# Graph builder
# ---------------------------

def build_graph(benefit_agent, claim_agent, summary_agent=None, ckpt_store=None):
    g = StateGraph(GraphState)

    g.add_node("router", router_node)

    def benefit_wrapper(state: GraphState) -> GraphState:
        try:
            logger.info(">>> Entered benefit_node with question=%s", state.question)
            res = benefit_agent.run(state.question, state.session_id, state.user_id)
            state.benefit_result = res["answer"]
            state.provenance += res.get("provenance", [])
            save_checkpoint(state, "benefit")
            logger.info(
                "After benefit_node start: route=%s original_route=%s needs_claim=%s",
                state.route, state.original_route, state.needs_claim
            )

            if state.needs_claim:
                logger.info("Routing to claim_node …")
                state = claim_node(state, claim_agent)

            state = summary_node(state)
            return state
        except Exception as e:
            logger.exception("Error inside benefit_wrapper: %s", e)
            raise

    g.add_node("benefit", benefit_wrapper)
    g.add_node("claim", lambda s: claim_node(s, claim_agent))
    g.add_node("summary_node", summary_node)
    g.add_node("noop", noop_node)

    g.set_entry_point("router")

    g.add_conditional_edges(
        "router",
        lambda s: s.route,
        {
            "benefit": "benefit",
            "claim": "claim",
            "both": "benefit",  # benefit wrapper handles claim+summary internally
            "clarify": END,
            "unknown": END,
        },
    )

    g.add_edge("claim", "summary_node")
    g.add_edge("summary_node", END)
    g.add_edge("benefit", "noop")
    g.add_edge("noop", END)

    return g.compile()


# ---------------------------
# Export a global graph placeholder
# ---------------------------

graph = None  # Will be set in main.py during startup


# ---------------------------
# Quick debug harness
# ---------------------------

if __name__ == "__main__":
    class DummyAgent:
        def run(self, q, sid, uid):
            logger.info("DummyAgent answering for %s", q)
            return {"answer": f"Answer for {q}", "provenance": [{"q": q}]}

    g = build_graph(DummyAgent(), DummyAgent())
    state = GraphState(session_id="s1", user_id="u1", question="Summarize benefits and claim status")
    final = g.invoke(state)
    print("Final summary:\n", final.summary)
    print("Provenance:\n", final.provenance)
    print("Checkpoint ID:", final.checkpoint_id)