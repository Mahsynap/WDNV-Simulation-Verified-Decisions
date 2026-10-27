"""Metrics for probabilistic diagnosis: calibration, credible sets, localisation distance."""
from __future__ import annotations
import networkx as nx
import numpy as np

TYPES = ("LEAK", "PUMP_FAILURE", "DEMAND_SURGE")


def ece(conf, correct, bins: int = 10) -> float:
    """Expected calibration error of top-1 confidence."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(tot)


def credible_set(probs: dict, mass: float = 0.8) -> list:
    out, acc = [], 0.0
    for k in sorted(probs, key=probs.get, reverse=True):
        out.append(k); acc += probs[k]
        if acc >= mass:
            break
    return out


class PipeGraph:
    """Hop distance between pipes (min over their end nodes) on the undirected network graph."""
    def __init__(self, wn):
        self.wn = wn
        self.G = wn.to_graph().to_undirected()
        self._cache = {}

    def _d(self, src):
        if src not in self._cache:
            self._cache[src] = nx.single_source_shortest_path_length(self.G, src)
        return self._cache[src]

    def hop(self, a: str, b: str) -> int:
        la, lb = self.wn.get_link(a), self.wn.get_link(b)
        best = 10 ** 6
        for x in (la.start_node_name, la.end_node_name):
            d = self._d(x)
            for y in (lb.start_node_name, lb.end_node_name):
                best = min(best, d.get(y, 10 ** 6))
        return best

    def node_hop(self, a: str, b: str) -> int:
        return self._d(a).get(b, 10 ** 6)


def true_hypothesis_index(lib, inc, pg: PipeGraph) -> int:
    """Library index that best represents the true incident (exact for leaks/pumps; nearest zone for surges)."""
    ids = [h.id for h in lib.hyps]
    if inc.kind == "LEAK":
        key = f"LEAK:{inc.params['pipe']}"
        if key in ids:
            return ids.index(key)
        cands = [(pg.hop(h.params["pipe"], inc.params["pipe"]), i) for i, h in enumerate(lib.hyps) if h.kind == "LEAK"]
        return min(cands)[1]
    if inc.kind == "PUMP_FAILURE":
        key = f"PUMP:{inc.params['pump']}"
        return ids.index(key) if key in ids else -1            # -1: the library has no hypothesis for this pump (callers skip)
    cands = [(pg.node_hop(h.params["center"], inc.params["center"]), i) for i, h in enumerate(lib.hyps) if h.kind == "DEMAND_SURGE"]
    return min(cands)[1]
