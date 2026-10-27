from __future__ import annotations
import numpy as np
import networkx as nx

from .config import Config
from .incidents import Incident, leak_names
from .network import Baseline, requested_demand


def supplied_mask(res, wn_run, base: Baseline, cfg: Config, cwin, eligible: list) -> np.ndarray:
    """[steps in window, eligible nodes] True where the node still has a hydraulic path to a source (tank / real reservoir).
    Nodes the action deliberately cut off are an *outage* (charged as unmet demand), not a pressure violation."""
    st, bst = res.link["status"], base.results.link["status"]
    cols = list(st.columns)
    S = (st.values[cwin] == 0)
    bidx = {c: i for i, c in enumerate(bst.columns)}
    Bv = (bst.values[cwin] == 0)
    Bm = np.zeros_like(S)
    for j, c in enumerate(cols):
        if c in bidx:
            Bm[:, j] = Bv[:, bidx[c]]
        if c.startswith(cfg.leak_prefix):
            S[:, j] = False                                       # the leak orifice itself is not a closure of the network
    extra = S & ~Bm                                               # closed now, open in the no-incident baseline
    mask = np.ones((S.shape[0], len(eligible)), dtype=bool)
    if not extra.any():
        return mask
    G, cache = None, {}
    for i in np.nonzero(extra.any(axis=1))[0]:
        closed = frozenset(cols[j] for j in np.nonzero(extra[i])[0])
        if closed not in cache:
            if G is None:
                G = wn_run.to_graph().to_undirected()
            H = G.copy()
            for u, v, k in list(H.edges(keys=True)):
                if k in closed:
                    H.remove_edge(u, v, k)
            src = [n for n in list(wn_run.tank_name_list) + list(wn_run.reservoir_name_list) if not n.startswith(cfg.sink_prefix) and n in H]
            reach = set()
            for sn in src:
                if sn not in reach:
                    reach |= nx.node_connected_component(H, sn)
            cache[closed] = np.array([n in reach for n in eligible])
        mask[i] = cache[closed]
    return mask


def compute_metrics(res, wn_run, base: Baseline, inc: Incident | None, n_ops: int, cfg: Config,
                    t_ref: int, t_act: int | None = None) -> dict:
    """Cost metrics (unmet demand, leak volume) cover [t_ref, horizon) = the whole incident.
    Constraint metrics cover [t_act, horizon): only the period an action can still influence.
    All are evaluated over the ORIGINAL junctions (the synthetic break node is excluded)."""
    t_act = t_ref if t_act is None else t_act
    horizon_s = wn_run.options.time.duration
    P_all = res.node["pressure"]
    times = P_all.index.values
    win = (times >= t_ref) & (times < horizon_s)
    dt = cfg.step_s
    J, E = base.junctions, base.eligible

    req = requested_demand(wn_run, times)[J].values[win]
    dem = res.node["demand"][J].values[win]
    unmet_m3 = float(np.clip(req - dem, 0.0, None).sum() * dt)
    req_m3 = float(req.sum() * dt)

    m = {"unmet_m3": unmet_m3, "requested_m3": req_m3,
         "service_availability": 1.0 - unmet_m3 / req_m3 if req_m3 > 0 else 1.0}

    cwin = (times >= t_act) & (times < horizon_s)                 # constraint window (post-action)
    PE = P_all[E].values[cwin]
    BPE = base.pressure[E].values[cwin]
    env_max = base.pressure[E].values.max(axis=0)                 # each node's own baseline maximum over the day
    M = supplied_mask(res, wn_run, base, cfg, cwin, E)                 # nodes still connected to a source
    m["n_nodes_cut_off"] = int((~M).any(axis=0).sum())
    m["min_pressure_eligible_m"] = float(np.where(M, PE, np.inf).min()) if M.any() else 100.0
    m["envelope_excess_m"] = float(np.where(M, PE - env_max[None, :], -np.inf).max()) if M.any() else 0.0
    m["n_nodes_below_hard_min"] = int(((PE < cfg.hard_min_pressure_m) & M).any(axis=0).sum())
    exc = (PE < cfg.service_pressure_m) & (BPE >= cfg.service_pressure_m) & M
    m["excess_low_pressure_share"] = float(exc.mean())

    # tanks: emptied relative to baseline behaviour
    emptied = 0
    min_margin = np.inf
    for tn in wn_run.tank_name_list:
        lv = P_all[tn].values[cwin]
        blv = base.results.node["pressure"][tn].values[(base.times >= t_act) & (base.times < horizon_s)]
        lo = wn_run.get_node(tn).min_level
        mm = float(lv.min() - lo)
        min_margin = min(min_margin, mm)
        if mm <= cfg.tank_margin_m and float(blv.min() - lo) > cfg.tank_margin_m:
            emptied += 1
    m["tanks_emptied"] = emptied
    m["min_tank_margin_m"] = float(min_margin) if np.isfinite(min_margin) else float("nan")

    leak_m3, reverse = 0.0, False
    if inc is not None and inc.kind == "LEAK":
        q = res.link["flowrate"][leak_names(inc, cfg)["link"]].values[win]
        leak_m3 = float(np.abs(q).sum() * dt)
        reverse = bool((q < -1e-6).any())
    m["leak_m3"] = leak_m3
    m["leak_reverse_flow"] = reverse
    m["ops"] = int(n_ops)
    m["objective_m3"] = cfg.w_unmet * unmet_m3 + cfg.w_leak * leak_m3 + cfg.ops_penalty_m3 * n_ops
    return m


def hard_violations(m: dict, cfg: Config) -> list:
    v = []
    if m["min_pressure_eligible_m"] < cfg.hard_min_pressure_m:
        v.append("min_pressure")
    if m["envelope_excess_m"] > cfg.hard_envelope_excess_m:
        v.append("pressure_envelope")
    if m["tanks_emptied"] > 0:
        v.append("tank_emptied")
    if m["leak_reverse_flow"]:
        v.append("model_artifact_reverse_leak_flow")
    return v


def severity(m: dict, cfg: Config) -> tuple:
    """Magnitude of hard-constraint violation (0 = none): used for the damage-limiting fallback."""
    return (max(0.0, cfg.hard_min_pressure_m - m["min_pressure_eligible_m"]),
            max(0.0, m["envelope_excess_m"] - cfg.hard_envelope_excess_m),
            float(m["tanks_emptied"]))
