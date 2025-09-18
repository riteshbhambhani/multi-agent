import pytest
from agents.orchestrator import _route

@pytest.mark.parametrize("q,route",[
    ("What's my copay for imaging?", "benefit"),
    ("Why was my claim denied?", "claim"),
    ("Does my plan cover ER and why was this claim denied?", "both"),
    ("hello", "clarify"),
])
def test_route(q, route):
    assert _route(q) == route
