from __future__ import annotations
import os, tempfile
from dataclasses import dataclass
import numpy as np
import pandas as pd
import wntr
from wntr.network import controls as C

from .config import Config

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "networks")


def resolve_path(name_or_path: str) -> str:
    if os.path.exists(name_or_path):
        return name_or_path
    lib = os.path.join(os.path.dirname(wntr.__file__), "library", "networks")
    for base in (_DATA, lib):
        for cand in (name_or_path, name_or_path + ".inp", name_or_path.lower() + ".inp", name_or_path.capitalize() + ".inp"):
            p = os.path.join(base, cand)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(name_or_path)


def standardize(wn, cfg: Config):
    """Common timing + PDD settings. Hourly patterns are repeated to the finer step."""
    t = wn.options.time
    k = int(round(t.pattern_timestep / cfg.step_s))
    if k > 1:
        for pn in wn.pattern_name_list:
            p = wn.get_pattern(pn)
            p.multipliers = np.repeat(np.asarray(p.multipliers, float), k)
    t.pattern_timestep = cfg.step_s
    t.hydraulic_timestep = cfg.step_s
    t.report_timestep = cfg.step_s
    t.duration = cfg.horizon_h * 3600
    t.start_clocktime = 0
    h = wn.options.hydraulic
    h.demand_model = "PDD"
    h.minimum_pressure = cfg.pdd_min_m
    h.required_pressure = cfg.pdd_req_m
    h.pressure_exponent = cfg.pdd_exp
    return wn


def load_network(name_or_path: str, cfg: Config):
    wn = wntr.network.WaterNetworkModel(resolve_path(name_or_path))
    return standardize(wn, cfg)


# --------------------------------------------------------------------------
# running + helpers
# --------------------------------------------------------------------------
def run_epanet(wn):
    with tempfile.TemporaryDirectory() as d:
        return wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(d, "w"))


def add_time_control(wn, t_s: int, link: str, status, name: str):
    act = C.ControlAction(wn.get_link(link), "status", status)
    wn.add_control(name, C.Control._time_control(wn, int(t_s), "SIM_TIME", False, act, name))


def drop_controls_for(wn, link_name: str) -> int:
    """Remove every control/rule whose action targets `link_name` (operator override)."""
    n = 0
    for cname, c in list(wn.controls()):
        hit = any(getattr(a.target()[0], "name", None) == link_name for a in c.actions())
        if hit:
            wn.remove_control(cname)
            n += 1
    return n


def limit_controls_before(wn, link_name: str, t0: int) -> dict:
    """Make every automatic control on `link_name` inactive from time t0 onwards, while leaving the
    trajectory before t0 untouched (time-scheduled controls firing >= t0 are removed; other controls
    are wrapped with an extra 'system time < t0' condition). Used for failures and operator overrides."""
    kept = removed = wrapped = 0
    for cname, c in list(wn.controls()):
        if not any(getattr(a.target()[0], "name", None) == link_name for a in c.actions()):
            continue
        cond = c.condition
        if isinstance(cond, C.SimTimeCondition):
            if cond._threshold >= t0:
                wn.remove_control(cname); removed += 1
            else:
                kept += 1
        else:
            c.update_condition(C.AndCondition(cond, C.SimTimeCondition(wn, "<", int(t0)))); wrapped += 1
    return {"kept": kept, "removed": removed, "wrapped": wrapped}


def requested_demand(wn, times) -> pd.DataFrame:
    """Requested (pre-PDD) demand per junction and time, m3/s."""
    t = wn.options.time
    idx = ((np.asarray(times, float) + t.pattern_start) // t.pattern_timestep).astype(int)
    mult = wn.options.hydraulic.demand_multiplier
    default_pat = wn.options.hydraulic.pattern
    cols = {}
    for jn in wn.junction_name_list:
        tot = np.zeros(len(idx))
        for ts in wn.get_node(jn).demand_timeseries_list:
            pname = ts.pattern_name if ts.pattern_name is not None else default_pat
            if pname is None or pname not in wn.pattern_name_list:
                m = np.ones(len(idx))
            else:
                arr = np.asarray(wn.get_pattern(pname).multipliers, float)
                m = arr[idx % len(arr)]
            tot = tot + ts.base_value * m
        cols[jn] = tot * mult
    return pd.DataFrame(cols, index=np.asarray(times, float))


@dataclass
class Baseline:
    wn: object
    results: object
    junctions: list
    pressure: pd.DataFrame
    eligible: list
    times: np.ndarray
    mean_demand_m3s: float


def compute_baseline(wn, cfg: Config) -> Baseline:
    res = run_epanet(wn)
    J = list(wn.junction_name_list)
    p = res.node["pressure"][J]
    ok = [j for j in J if p[j].min() >= cfg.hard_min_pressure_m + cfg.eligible_margin_m]
    req = requested_demand(wn, p.index.values)
    return Baseline(wn=wn, results=res, junctions=J, pressure=p, eligible=ok,
                    times=p.index.values, mean_demand_m3s=float(req.sum(axis=1).mean()))
