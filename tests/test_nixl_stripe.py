"""scripts/nixl_patch/kvtb_stripe.py against a fake nixl_agent that records where each block lands."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "nixl_patch"))
import kvtb_stripe  # noqa: E402

FABRIC: dict[str, "FakeAgent"] = {}  # agent name -> agent, so a WRITE can reach its peer


class FakeAgent:
    def __init__(self, name, config=None):
        self.name, self.remotes, self.landed, self.backends = name, {}, {}, []
        self.fail_check = False
        FABRIC[name] = self

    def create_backend(self, backend, params):
        self.backends.append((backend, params))

    def register_memory(self, reg_list, mem_type=None, backends=()):
        return ("reg", self.name)

    def deregister_memory(self, reg, backends=()):
        assert reg == ("reg", self.name)

    def get_xfer_descs(self, descs, mem_type=None):
        return list(descs)

    def prep_xfer_dlist(self, agent_name, descs, mem_type=None, backends=()):
        owner = self.name if agent_name == "NIXL_INIT_AGENT" else self._name(agent_name)
        return {"owner": owner, "descs": descs}

    def release_dlist_handle(self, h):
        h["released"] = True

    def get_agent_metadata(self):
        return self.name.encode()

    def add_remote_agent(self, md):
        name = md.decode()
        self.remotes[name] = FABRIC[name]
        return name.encode()  # the real nixl_agent returns bytes

    def _name(self, name):
        return name.decode() if isinstance(name, bytes) else name

    def remove_remote_agent(self, name):
        del self.remotes[self._name(name)]

    def make_prepped_xfer(self, op, local, li, remote, ri, *a, **kw):
        assert local["owner"] == self.name and remote["owner"] in self.remotes, "stripe crossed agents"
        return {
            "op": op,
            "local": [int(x) for x in li],
            "remote": [int(x) for x in ri],
            "to": remote["owner"],
            "polls": 0,
            "released": False,
        }

    def transfer(self, h, notif=b""):
        return "PROC"

    def check_xfer_state(self, h):
        if self.fail_check:
            return "ERR"
        h["polls"] += 1
        if h["polls"] < 2:
            return "PROC"
        for j, r in zip(h["local"], h["remote"]):
            FABRIC[h["to"]].landed[r] = j
        return "DONE"

    def release_xfer_handle(self, h):
        h["released"] = True


def _pair(k):
    FABRIC.clear()
    a = kvtb_stripe.factory(FakeAgent, k)("A")
    b = kvtb_stripe.factory(FakeAgent, k)("B")
    for s in (a, b):
        s.create_backend("UCX", {"num_threads": "2"})
        s.register_memory([(0, 1, 0, "")], "DRAM")
    name = a.add_remote_agent(b.get_agent_metadata())
    local = a.prep_xfer_dlist("NIXL_INIT_AGENT", a.get_xfer_descs([(0, 1, 0)]))
    remote = a.prep_xfer_dlist(name, a.get_xfer_descs([(0, 1, 0)]))
    return a, b, name, local, remote


def test_stripe_bounds_cover_every_index_once():
    for n in (0, 1, 7, 37, 292):
        for k in (1, 3, 8, 16):
            idx = [i for s, e in kvtb_stripe.stripe_bounds(n, k) for i in range(s, e)]
            assert idx == list(range(n))
            assert len(kvtb_stripe.stripe_bounds(n, k)) == min(n, k)


def test_metadata_round_trip_and_rejects_unstriped_peer():
    blobs = [b"x", b"", b"\x00" * 300]
    assert kvtb_stripe.unpack_metadata(kvtb_stripe.pack_metadata(blobs)) == blobs
    with pytest.raises(ValueError, match="not striped"):
        kvtb_stripe.unpack_metadata(b"plain nixl metadata")


@pytest.mark.parametrize("k", [1, 4, 8])
def test_every_block_lands_at_its_paired_remote_index(k):
    a, b, _, local, remote = _pair(k)
    li = np.arange(37, dtype=np.int32)
    ri = (li * 7 + 5) % 101  # non-contiguous, so a mispaired stripe would show
    h = a.make_prepped_xfer("WRITE", local, li, remote, ri)
    assert len(h.parts) == min(k, 37)
    assert a.transfer(h) == "PROC"
    while (state := a.check_xfer_state(h)) == "PROC":
        pass
    assert state == "DONE"
    landed = {r: j for agent in b.agents for r, j in agent.landed.items()}
    assert landed == {int(r): int(j) for j, r in zip(li, ri)}
    # agent i of A only ever talks to agent i of B
    assert all(p["to"] == f"B-s{i}" for i, p in h.parts)
    a.release_xfer_handle(h)
    assert all(p["released"] for _, p in h.parts)


def test_backend_params_reach_every_agent_and_err_is_reported():
    a, _, name, local, remote = _pair(4)
    assert all(ag.backends == [("UCX", {"num_threads": "2"})] for ag in a.agents)
    h = a.make_prepped_xfer("WRITE", local, [0, 1, 2, 3], remote, [4, 5, 6, 7])
    a.agents[2].fail_check = True
    assert a.check_xfer_state(h) == "ERR"
    a.remove_remote_agent(name)
    assert all(not ag.remotes for ag in a.agents)


def test_stripe_count_mismatch_is_refused():
    FABRIC.clear()
    a = kvtb_stripe.factory(FakeAgent, 4)("A")
    b = kvtb_stripe.factory(FakeAgent, 8)("B")
    with pytest.raises(ValueError, match="stripes"):
        a.add_remote_agent(b.get_agent_metadata())


def test_async_post_reports_proc_until_the_post_returns_and_surfaces_its_error():
    import threading

    gate = threading.Event()

    class SlowPost(FakeAgent):
        def transfer(self, h, notif=b""):
            gate.wait(5)
            if self.name == "A-s1":
                raise RuntimeError("nixlRemoteDisconnectError")
            return "PROC"

    FABRIC.clear()
    a = kvtb_stripe.factory(SlowPost, 2)("A")
    b = kvtb_stripe.factory(SlowPost, 2)("B")
    name = a.add_remote_agent(b.get_agent_metadata())
    local = a.prep_xfer_dlist("NIXL_INIT_AGENT", [])
    remote = a.prep_xfer_dlist(name, [])
    h = a.make_prepped_xfer("WRITE", local, [0, 1], remote, [2, 3])
    assert a.transfer(h) == "PROC"  # returns while both posts are blocked
    assert a.check_xfer_state(h) == "PROC"  # posts still running: nothing touched the agents yet
    assert all(p["polls"] == 0 for _, p in h.parts)
    gate.set()
    h.posts[1].exception()
    with pytest.raises(RuntimeError, match="nixlRemoteDisconnectError"):
        while a.check_xfer_state(h) == "PROC":
            pass
    a.release_xfer_handle(h)  # waits for the posts, then releases every part
    assert all(p["released"] for _, p in h.parts)
