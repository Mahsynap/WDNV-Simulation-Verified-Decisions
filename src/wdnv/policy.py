"""Decision policies that turn diagnosis beliefs into actions (with or without simulation-based verification)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .config import Config
from .actions import Action, HOLD, enumerate_candidates
from .incidents import Incident
from .metrics import severity
from .verifier import Model, Outcome, evaluate, model_with_incident


def candidate_pool(twin: Model, beliefs: list, cfg: Config) -> list:
    pool, seen = [], set()
    for inc, _ in beliefs:
        for a in enumerate_candidates(model_with_incident(twin, inc, cfg), inc, cfg):
            if a.key() not in seen:
                seen.add(a.key()); pool.append(a)
    if HOLD.key() not in seen:
        pool.insert(0, HOLD)
    return pool


@dataclass
class Table:
    """objective/safety of every candidate under every belief, evaluated on the twin."""
    pool: list
    weights: np.ndarray
    outs: list                     # outs[a][j] -> Outcome

    def expected(self, a: int) -> float:
        return float(sum(w * o.objective for w, o in zip(self.weights, self.outs[a])))

    def ok(self, a: int) -> bool:
        return all(o.status == "ok" for o in self.outs[a])

    def safe(self, a: int, p_min: float) -> bool:
        return all(o.hard_safe for w, o in zip(self.weights, self.outs[a]) if w >= p_min)

    def regret(self, a: int) -> float:
        """expected regret of candidate a: sum_j w_j (J_j(a) - min_a' J_j(a'))"""
        r = 0.0
        for j, w in enumerate(self.weights):
            best = min(self.outs[b][j].objective for b in range(len(self.pool)) if self.outs[b][j].status == "ok")
            r += w * (self.outs[a][j].objective - best)
        return float(r)


def build_table(twin: Model, beliefs: list, t_action: int, cfg: Config) -> Table:
    pool = candidate_pool(twin, beliefs, cfg)
    w = np.array([b[1] for b in beliefs], float)
    outs = [[evaluate(twin, inc, a, t_action, cfg) for inc, _ in beliefs] for a in pool]
    return Table(pool, w, outs)


def select_verified(tab: Table, cfg: Config, p_min: float = 0.10):
    """Robust verified choice. Returns (action index or None, mode)."""
    hold = [i for i, a in enumerate(tab.pool) if a.kind == "HOLD"][0]
    cands = [i for i in range(len(tab.pool)) if tab.ok(i) and tab.pool[i].executable()]
    safe = sorted([i for i in cands if tab.safe(i, p_min)], key=tab.expected)
    if safe:
        return safe[0], "safe_best"
    if tab.ok(hold):
        hs = [severity(o.metrics, cfg) for o in tab.outs[hold]]
        dl = []
        for i in cands:
            if tab.expected(i) < tab.expected(hold) and all(all(x <= y + 1e-9 for x, y in zip(severity(o.metrics, cfg), h))
                                                           for o, h in zip(tab.outs[i], hs)):
                dl.append(i)
        if dl:
            return sorted(dl, key=tab.expected)[0], "damage_limiting"
    return None, "review"


def naive_response(top: Incident, twin: Model, cfg: Config) -> Action:
    """Un-verified 'playbook' reaction to the single most likely incident."""
    if top.kind == "LEAK":
        return Action("CLOSE_PIPES", {"pipes": [top.params["pipe"]]})
    if top.kind == "PUMP_FAILURE":
        others = [p for p in twin.wn.pump_name_list if p != top.params["pump"]]
        return Action("SET_PUMP_STATUS", {"pump": others[0], "status": "OPEN"}) if others else HOLD
    return HOLD
