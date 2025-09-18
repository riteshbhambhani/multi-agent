from ..agents import ckpt_store

def test_ckpt_cycle(tmp_path, monkeypatch):
    # Use real DB per env
    res = ckpt_store.create("u1","s1","claim","Provide date","{}")
    ck = ckpt_store.get(res["checkpoint_id"])
    assert ck and ck["pending_agent"]=="claim"
    ckpt_store.delete(res["checkpoint_id"])
    assert ckpt_store.get(res["checkpoint_id"]) is None
