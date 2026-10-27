from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import networkx as nx
import wntr
from wntr.network.elements import LinkStatus

from .config import Config
from .network import add_time_control, limit_controls_before, Baseline

G_ACC = 9.81
KINDS = ("LEAK", "PUMP_FAILURE", "DEMAND_SURGE")


@dataclass
class Incident:
    id: str
    kind: str
    t_start: int                       # seconds from simulation start
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "t_start": int(self.t_start), "params": self.params}

    @staticmethod
    def from_dict(d: dict) -> "Incident":
        return Incident(d["id"], d["kind"], int(d["t_start"]), dict(d.get("params", {})))


def leak_names(inc: Incident, cfg: Config) -> dict:
    pipe = inc.params["pipe"]
    return {"node": f"{cfg.break_prefix}{pipe}", "pipe_a": pipe, "pipe_b": f"{pipe}__b",
            "sink": f"{cfg.sink_prefix}{inc.id}", "link": f"{cfg.leak_prefix}{inc.id}"}


# --------------------------------------------------------------------------
# applying an incident to a model (mutates wn)
# --------------------------------------------------------------------------
def apply_incident(wn, inc: Incident, cfg: Config):
    if inc.kind == "LEAK":
        _apply_leak(wn, inc, cfg)
    elif inc.kind == "PUMP_FAILURE":
        pump = inc.params["pump"]
        limit_controls_before(wn, pump, inc.t_start)          # nothing may re-open the failed pump
        add_time_control(wn, inc.t_start, pump, LinkStatus.Closed, f"fail_{inc.id}")
    elif inc.kind == "DEMAND_SURGE":
        _apply_surge(wn, inc, cfg)
    else:
        raise ValueError(f"unknown incident kind {inc.kind}")
    return wn


def _apply_leak(wn, inc, cfg):
    nm = leak_names(inc, cfg)
    wntr.morph.split_pipe(wn, nm["pipe_a"], nm["pipe_b"], nm["node"], return_copy=False)
    node = wn.get_node(nm["node"])
    area = float(inc.params["area"])
    cd = float(inc.params.get("cd", cfg.leak_cd))
    d = float(np.sqrt(4.0 * area / np.pi))
    wn.add_reservoir(nm["sink"], base_head=node.elevation)           # atmospheric sink
    # orifice: head loss = (1/Cd^2) v^2/2g with pipe area = orifice area; friction removed (C huge)
    wn.add_pipe(nm["link"], nm["node"], nm["sink"], length=0.1, diameter=d, roughness=1e5,
                minor_loss=1.0 / cd ** 2, initial_status="Closed")
    add_time_control(wn, inc.t_start, nm["link"], LinkStatus.Open, f"open_{nm['link']}")
    if inc.params.get("t_end"):
        add_time_control(wn, int(inc.params["t_end"]), nm["link"], LinkStatus.Closed, f"close_{nm['link']}")


def _apply_surge(wn, inc, cfg):
    p = inc.params
    factor = float(p["factor"])
    t0, t1 = int(inc.t_start), int(inc.t_start + p["duration"])
    step = wn.options.time.pattern_timestep
    nsteps = int(wn.options.time.duration // step) + 2
    mask = np.array([1.0 if t0 <= k * step < t1 else 0.0 for k in range(nsteps)])
    default_pat = wn.options.hydraulic.pattern
    cache = {}
    for jn in p["nodes"]:
        j = wn.get_node(jn)
        ts0 = j.demand_timeseries_list[0]
        pname = ts0.pattern_name if ts0.pattern_name is not None else default_pat
        if pname not in cache:
            arr = np.asarray(wn.get_pattern(pname).multipliers, float) if (pname in wn.pattern_name_list) else np.ones(1)
            new = f"SRG_{inc.id}_{len(cache)}"
            wn.add_pattern(new, (np.resize(arr, nsteps) * mask).tolist())
            cache[pname] = new
        j.add_demand(ts0.base_value * (factor - 1.0), cache[pname], category="surge")


# --------------------------------------------------------------------------
# sampling incidents
# --------------------------------------------------------------------------
def _grid_time(rng, lo_h, hi_h, step):
    ks = np.arange(int(lo_h * 3600 // step), int(hi_h * 3600 // step) + 1)
    return int(rng.choice(ks) * step)


def sample_leak(base: Baseline, cfg: Config, rng, idx: int):
    wn = base.wn
    elig = set(base.eligible)
    pipes = [n for n, p in wn.pipes()
             if p.start_node_name in elig and p.end_node_name in elig and not p.check_valve
             and p.initial_status == LinkStatus.Open and p.length >= 20.0]
    if not pipes:
        return None
    pipe = str(rng.choice(pipes))
    t0 = _grid_time(rng, 2, 12, cfg.step_s)
    p_est = float(base.pressure[[wn.get_link(pipe).start_node_name, wn.get_link(pipe).end_node_name]].mean().mean())
    frac = float(rng.uniform(*cfg.leak_frac_range))
    q = frac * base.mean_demand_m3s
    area = q / (cfg.leak_cd * np.sqrt(2 * G_ACC * max(p_est, 5.0)))
    return Incident(f"LEA{idx:03d}", "LEAK", t0, {"pipe": pipe, "area": round(float(area), 7), "cd": cfg.leak_cd,
                                                   "target_frac_of_demand": round(frac, 4)})


def sample_pump_failure(base: Baseline, cfg: Config, rng, idx: int):
    q = base.results.link["flowrate"]
    pumps = [p for p in base.wn.pump_name_list if (q[p].abs() > 1e-6).any()]
    if not pumps:
        return None
    for _ in range(50):
        pump = str(rng.choice(pumps))
        t0 = _grid_time(rng, 2, 14, cfg.step_s)
        if abs(float(q[pump].iloc[int(t0 // cfg.step_s)])) > 1e-6:      # pump actually running when it fails
            return Incident(f"PUM{idx:03d}", "PUMP_FAILURE", t0, {"pump": pump})
    return None


def sample_surge(base: Baseline, cfg: Config, rng, idx: int):
    wn = base.wn
    G = wn.to_graph().to_undirected()
    dem = {j: wn.get_node(j).demand_timeseries_list[0].base_value for j in base.junctions}
    pos = [j for j in base.junctions if dem[j] > 0]
    if not pos:
        return None
    total = sum(dem.values())
    target = float(rng.uniform(0.05, 0.15)) * total
    center = str(rng.choice(pos))
    nodes, acc = [], 0.0
    for n in nx.bfs_tree(G, center):
        if n in dem and dem[n] > 0:
            nodes.append(n)
            acc += dem[n]
            if acc >= target or len(nodes) >= 150:
                break
    t0 = _grid_time(rng, 2, 14, cfg.step_s)
    dur = int(rng.choice([2, 3, 4, 6]) * 3600)
    return Incident(f"SUR{idx:03d}", "DEMAND_SURGE", t0, {"nodes": nodes, "factor": round(float(rng.uniform(1.5, 3.0)), 3),
                                                         "duration": dur, "center": center})


def generate_incidents(base: Baseline, cfg: Config, rng, counts: dict) -> list:
    fn = {"LEAK": sample_leak, "PUMP_FAILURE": sample_pump_failure, "DEMAND_SURGE": sample_surge}
    out = []
    for kind, n in counts.items():
        for i in range(n):
            inc = fn[kind](base, cfg, rng, i)
            if inc is not None:
                out.append(inc)
    return out
