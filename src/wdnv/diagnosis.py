"""Diagnosis from telemetry: detect -> score hypotheses against a signature library -> calibrated beliefs.

Classical and transparent on purpose (residual detector + sensitivity-signature matching). It is the baseline
and the source of *beliefs* (possibly wrong) that the verifier/agent layers must cope with."""
from __future__ import annotations
from dataclasses import dataclass, field
import networkx as nx
import numpy as np
import pandas as pd

from .config import Config
from .incidents import Incident, G_ACC
from .network import Baseline
from .telemetry import SensorSet
from .verifier import Model, model_with_incident, run_epanet
from .sync import SyncTwin

TYPES = ("LEAK", "PUMP_FAILURE", "DEMAND_SURGE")


# ---------------------------------------------------------------------------------------------
# signature library
# ---------------------------------------------------------------------------------------------
@dataclass
class Hypothesis:
    id: str
    kind: str
    params: dict
    t_ref: int
    amp_unit: float            # leak: reference orifice area (m2); surge: factor-1 (=1); pump: 1


@dataclass
class SignatureLibrary:
    hyps: list
    P: np.ndarray              # [n_h, n_junctions]  pressure change (m), unit amplitude
    Q: np.ndarray              # [n_h, n_pumps]      pump-flow change (L/s)
    L: np.ndarray              # [n_h, n_tanks]      tank-level change (m)
    junctions: list
    pumps: list
    tanks: list
    window: int
    offset: int

    def matrix(self, s: SensorSet) -> np.ndarray:
        """Signature matrix [n_h, channels] with channel order P:* , L:* , S:* (pump status gap).
        Pump-flow residuals are deliberately NOT used (flat pump curves make them hypersensitive to model error)."""
        ji = [self.junctions.index(n) for n in s.pressure_nodes]
        li = [self.tanks.index(n) for n in s.tank_nodes]
        S = np.zeros((len(self.hyps), len(s.flow_links)))
        for i, h in enumerate(self.hyps):
            if h.kind == "PUMP_FAILURE" and h.params["pump"] in s.flow_links:
                S[i, s.flow_links.index(h.params["pump"])] = -1.0         # observed off while controls say on
        return np.hstack([self.P[:, ji], self.L[:, li], S]).astype(float)

    def kinds(self) -> np.ndarray:
        return np.array([h.kind for h in self.hyps])


def _lib_pipes(wn):
    J = set(wn.junction_name_list)
    from wntr.network.elements import LinkStatus
    return [n for n, p in wn.pipes() if p.start_node_name in J and p.end_node_name in J
            and not p.check_valve and p.initial_status == LinkStatus.Open and p.length >= 20.0]


def build_library(twin: Model, cfg: Config, rng, window: int = 4, offset: int = 1, leak_ref_h: float = 6.0,
                  leak_ref_frac: float = 0.05, n_zones: int = 40, surge_share: float = 0.10) -> SignatureLibrary:
    wn0, base, dt = twin.wn, twin.base, cfg.step_s
    J, pumps, tanks = base.junctions, list(wn0.pump_name_list), list(wn0.tank_name_list)
    bP = base.pressure[J].values
    bQ = base.results.link["flowrate"][pumps].values * 1000.0 if pumps else np.zeros((len(base.times), 0))
    bL = base.results.node["pressure"][tanks].values if tanks else np.zeros((len(base.times), 0))

    def signature(res, t_ref):
        k0 = int(t_ref // dt) + offset
        ix = slice(k0, k0 + window)
        P = res.node["pressure"][J].values[ix].mean(0) - bP[ix].mean(0)
        Q = (res.link["flowrate"][pumps].values[ix].mean(0) * 1000.0 - bQ[ix].mean(0)) if pumps else np.zeros(0)
        L = (res.node["pressure"][tanks].values[ix].mean(0) - bL[ix].mean(0)) if tanks else np.zeros(0)
        return P, Q, L

    def simulate(inc, t_ref):
        wn = model_with_incident(twin, inc, cfg)
        wn.options.time.duration = int(t_ref + (offset + window + 1) * dt)
        return run_epanet(wn)

    hyps, Ps, Qs, Ls = [], [], [], []

    def add(h, sig):
        hyps.append(h); Ps.append(sig[0]); Qs.append(sig[1]); Ls.append(sig[2])

    # ---- leaks: every eligible-or-not pipe between two junctions
    t_ref = int(leak_ref_h * 3600 // dt * dt)
    k_on = t_ref // dt
    for pipe in _lib_pipes(wn0):
        l = wn0.get_link(pipe)
        p_est = float(base.pressure[[l.start_node_name, l.end_node_name]].values[k_on].mean())
        if p_est < 8.0:
            continue
        area = leak_ref_frac * base.mean_demand_m3s / (cfg.leak_cd * np.sqrt(2 * G_ACC * p_est))
        inc = Incident("LIB", "LEAK", t_ref, {"pipe": pipe, "area": float(area), "cd": cfg.leak_cd})
        try:
            add(Hypothesis(f"LEAK:{pipe}", "LEAK", {"pipe": pipe}, t_ref, float(area)), signature(simulate(inc, t_ref), t_ref))
        except Exception:
            continue
    # ---- pump failures (at a time the pump is running)
    qb = base.results.link["flowrate"]
    for pump in pumps:
        ks = [k for k in range(int(4 * 3600 // dt), int(20 * 3600 // dt)) if abs(float(qb[pump].iloc[k])) > 1e-6]
        if not ks:
            continue
        tp = ks[0] * dt
        inc = Incident("LIB", "PUMP_FAILURE", tp, {"pump": pump})
        add(Hypothesis(f"PUMP:{pump}", "PUMP_FAILURE", {"pump": pump}, tp, 1.0), signature(simulate(inc, tp), tp))
    # ---- demand surges by zone
    G = wn0.to_graph().to_undirected()
    dem = {j: wn0.get_node(j).demand_timeseries_list[0].base_value for j in J}
    pos = [j for j in J if dem[j] > 0]
    total = sum(dem.values())
    centers = [str(rng.choice(pos))]
    dist = nx.single_source_shortest_path_length(G, centers[0])
    while len(centers) < min(n_zones, len(pos)):
        nxt = max(pos, key=lambda n: dist.get(n, 0))
        if dist.get(nxt, 0) == 0:
            break
        centers.append(nxt)
        d2 = nx.single_source_shortest_path_length(G, nxt)
        dist = {n: min(dist.get(n, 1e9), d2.get(n, 1e9)) for n in set(dist) | set(d2)}
    for c in centers:
        nodes, acc = [], 0.0
        for n in nx.bfs_tree(G, c):
            if n in dem and dem[n] > 0:
                nodes.append(n); acc += dem[n]
                if acc >= surge_share * total or len(nodes) >= 150:
                    break
        inc = Incident("LIB", "DEMAND_SURGE", t_ref, {"nodes": nodes, "factor": 2.0, "duration": 12 * 3600, "center": c})
        add(Hypothesis(f"SURGE:{c}", "DEMAND_SURGE", {"center": c, "nodes": nodes}, t_ref, 1.0), signature(simulate(inc, t_ref), t_ref))
    return SignatureLibrary(hyps, np.array(Ps), np.array(Qs), np.array(Ls), J, pumps, tanks, window, offset)


# ---------------------------------------------------------------------------------------------
# alarm-conditioned features and noise model
# ---------------------------------------------------------------------------------------------
@dataclass
class WindowSpec:
    """Features are (post - pre) differences around an alarm at step k_alarm, so slow model mismatch cancels."""
    pre_lag: int = 5          # pre window ends this many steps before the alarm (alarm latency must be < pre_lag steps)
    pre_len: int = 6
    post_lead: int = 1        # post window starts this many steps before the alarm ...
    wait: int = 3             # ... and ends this many steps after it (decision waits `wait` steps)


def channel_names(s: SensorSet) -> list:
    return [f"P:{n}" for n in s.pressure_nodes] + [f"L:{n}" for n in s.tank_nodes] + [f"S:{l}" for l in s.flow_links]


def features(tel: pd.DataFrame, exp: pd.DataFrame, gap: pd.DataFrame, s: SensorSet, k_alarm: int, w: WindowSpec) -> np.ndarray:
    pl = [c for c in tel.columns if c.startswith(("P:", "L:"))]
    r = (tel[pl] - exp[pl]).values
    post = np.nanmean(r[max(0, k_alarm - w.post_lead): k_alarm + w.wait + 1], axis=0)
    hi = max(1, k_alarm - w.pre_lag + 1)
    pre = np.nanmean(r[max(0, hi - w.pre_len): hi], axis=0)
    d_pl = post - pre
    g = np.array([np.nanmean(gap[l].values[max(0, k_alarm - w.post_lead): k_alarm + w.wait + 1]) if l in gap.columns else 0.0
                  for l in s.flow_links])
    return np.concatenate([d_pl, g])


@dataclass
class NoiseModel:
    sigma: np.ndarray

    @staticmethod
    def fit(sync: SyncTwin, sensors: SensorSet, normal_tels: list, w: WindowSpec, rng, per_day: int = 6,
            k_range=(12, 80)) -> "NoiseModel":
        feats = []
        for t in normal_tels:
            exp, gap = sync.expected(t), sync.status_gap(t)
            for k in rng.integers(k_range[0], k_range[1], size=per_day):
                feats.append(features(t, exp, gap, sensors, int(k), w))
        F = np.array(feats)
        sd = np.nanstd(F, axis=0)
        names = channel_names(sensors)
        floor = np.array([0.3 if n.startswith("P:") else 0.03 if n.startswith("L:") else 0.1 for n in names])
        return NoiseModel(np.maximum(sd, floor))


# ---------------------------------------------------------------------------------------------
# diagnosis
# ---------------------------------------------------------------------------------------------
@dataclass
class Diagnosis:
    chi2: np.ndarray                         # [n_h + 1]; last entry = null hypothesis (nothing happened)
    amp: np.ndarray                          # [n_h] fitted amplitudes
    d_obs: np.ndarray
    k_alarm: int

    def posterior(self, lib: SignatureLibrary, T: float, null_prior: float = 0.05) -> np.ndarray:
        kinds = lib.kinds()
        prior = np.zeros(len(kinds) + 1)
        for t in TYPES:
            m = kinds == t
            if m.any():
                prior[:-1][m] = (1 - null_prior) / len(TYPES) / m.sum()
        prior[-1] = null_prior
        lg = -self.chi2 / (2.0 * T) + np.log(np.maximum(prior, 1e-300))
        lg -= lg.max()
        p = np.exp(lg)
        return p / p.sum()

    def type_probs(self, lib, T) -> dict:
        p = self.posterior(lib, T); kinds = lib.kinds()
        out = {t: float(p[:-1][kinds == t].sum()) for t in TYPES}
        out["NONE"] = float(p[-1])
        return out

    def leak_probs(self, lib, T) -> dict:
        """Distribution over leak pipes conditional on a leak (own temperature, see fit_temperature)."""
        m = lib.kinds() == "LEAK"
        c = self.chi2[:-1][m]
        lg = -c / (2.0 * T); lg -= lg.max()
        p = np.exp(lg); p /= p.sum()
        return {h.params["pipe"]: float(pi) for h, pi in zip([h for h in lib.hyps if h.kind == "LEAK"], p)}


class Diagnoser:
    AMP_MAX = {"LEAK": 6.0, "PUMP_FAILURE": 1.5, "DEMAND_SURGE": 3.0}

    def __init__(self, lib: SignatureLibrary, sensors: SensorSet, sync: SyncTwin, noise: NoiseModel, win: WindowSpec | None = None):
        self.lib, self.sensors, self.sync, self.noise = lib, sensors, sync, noise
        self.win = win or WindowSpec()
        self.S = lib.matrix(sensors)
        self.T_type, self.T_loc = 1.0, 1.0
        self.amp_max = np.array([self.AMP_MAX[h.kind] for h in lib.hyps])

    def score(self, d: np.ndarray):
        w = 1.0 / self.noise.sigma ** 2
        ok = ~np.isnan(d)
        d0, S, w = np.where(ok, d, 0.0), np.where(ok[None, :], self.S, 0.0), np.where(ok, w, 0.0)
        a = np.clip((S * (w * d0)[None, :]).sum(1) / np.maximum((S ** 2 * w[None, :]).sum(1), 1e-12), 0.0, self.amp_max)
        chi = (((d0[None, :] - a[:, None] * S) ** 2) * w[None, :]).sum(1)
        return np.append(chi, float((d0 ** 2 * w).sum())), a, d0

    def diagnose(self, tel: pd.DataFrame, k_alarm: int, exp: pd.DataFrame | None = None, gap: pd.DataFrame | None = None) -> Diagnosis:
        exp = self.sync.expected(tel) if exp is None else exp
        gap = self.sync.status_gap(tel) if gap is None else gap
        d = features(tel, exp, gap, self.sensors, k_alarm, self.win)
        chi, a, d0 = self.score(d)
        return Diagnosis(chi, a, d0, k_alarm)

    def beliefs(self, dg: Diagnosis, cfg: Config, top_k: int = 3, T: float | None = None) -> list:
        """Top-k (Incident belief, probability) pairs from the joint posterior."""
        T = self.T_type if T is None else T
        p = dg.posterior(self.lib, T)
        order = [i for i in np.argsort(-p) if i < len(self.lib.hyps)][:top_k]
        t_est = max(0, int((dg.k_alarm - 2) * cfg.step_s))
        out = []
        for i in order:
            h, a = self.lib.hyps[i], float(dg.amp[i])
            if h.kind == "LEAK":
                inc = Incident("BEL", "LEAK", t_est, {"pipe": h.params["pipe"], "area": max(a, 0.05) * h.amp_unit, "cd": cfg.leak_cd})
            elif h.kind == "PUMP_FAILURE":
                inc = Incident("BEL", "PUMP_FAILURE", t_est, {"pump": h.params["pump"]})
            else:
                inc = Incident("BEL", "DEMAND_SURGE", t_est, {"nodes": h.params["nodes"], "factor": 1.0 + max(a, 0.05),
                                                              "duration": 6 * 3600, "center": h.params["center"]})
            out.append((inc, float(p[i])))
        return out


def belief_set(dg: "Diagnoser", d: Diagnosis, cfg: Config, per_type: int = 2, min_type_p: float = 0.10) -> list:
    """Type-aware belief set: for every incident type with posterior >= min_type_p keep its `per_type` best hypotheses.
    Weights = type probability x within-type probability (softmax with the localisation temperature), renormalised."""
    lib = dg.lib
    tp = d.type_probs(lib, dg.T_type)
    kinds = lib.kinds()
    t_est = max(0, int((d.k_alarm - 2) * cfg.step_s))
    out = []
    for t in TYPES:
        if tp[t] < min_type_p:
            continue
        idx = np.where(kinds == t)[0]
        lg = -d.chi2[idx] / (2.0 * dg.T_loc); lg -= lg.max()
        w = np.exp(lg); w /= w.sum()
        for j in np.argsort(-w)[:per_type]:
            i = int(idx[j]); h, a = lib.hyps[i], float(d.amp[i])
            if t == "LEAK":
                inc = Incident("BEL", "LEAK", t_est, {"pipe": h.params["pipe"], "area": max(a, 0.05) * h.amp_unit, "cd": cfg.leak_cd})
            elif t == "PUMP_FAILURE":
                inc = Incident("BEL", "PUMP_FAILURE", t_est, {"pump": h.params["pump"]})
            else:
                inc = Incident("BEL", "DEMAND_SURGE", t_est, {"nodes": h.params["nodes"], "factor": 1.0 + max(a, 0.05),
                                                              "duration": 6 * 3600, "center": h.params["center"]})
            out.append((inc, tp[t] * float(w[j])))
    z = sum(w for _, w in out) or 1.0
    return [(i, w / z) for i, w in out]


def fit_temperature(chi_list: list, true_idx: list, grid=None) -> float:
    """Temperature minimising the NLL of the true hypothesis under softmax(-chi2/2T)."""
    grid = np.logspace(-1, 3.5, 60) if grid is None else grid
    best, bT = np.inf, 1.0
    for T in grid:
        nll = 0.0
        for chi, t in zip(chi_list, true_idx):
            lg = -chi / (2 * T); lg -= lg.max()
            nll -= lg[t] - np.log(np.exp(lg).sum())
        if nll < best:
            best, bT = nll, float(T)
    return bT
