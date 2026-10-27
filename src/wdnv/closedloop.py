"""Closed-loop staged response: probe (close the boundary of a region) -> observe -> re-diagnose -> ... -> isolate.

Each probe is a real, executable action with a real outage cost; the posterior over leak locations is updated by
re-simulating the leading hypotheses under the *whole action history* on the synchronised twin."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .actions import Action, HOLD
from .config import Config
from .diagnosis import Diagnoser, WindowSpec, features
from .incidents import Incident, G_ACC
from .policy import build_table, candidate_pool, select_verified
from .regions import RegionIndex


@dataclass
class ClosedLoopParams:
    max_stages: int = 3
    stage_gap_steps: int = 4          # time between stage actions (1 h)
    observe_steps: int = 2            # steps observed after an action before updating
    top_m: int = 20                   # hypotheses re-simulated per update
    stop_conf: float = 0.6            # stop probing once the top pipe has this posterior
    value_scale: float = 1.0          # multiplies the information value of a probe (sensitivity parameter)
    force: bool = False               # ablation: probe even when the expected value is negative
    T_probe: float = 1.0              # temperature of the probe likelihood
    max_boundary: int = 12
    min_leak_prob: float = 0.5        # below this the incident is not treated as a leak (single-shot fallback)


PROBE_WIN = WindowSpec(pre_lag=1, pre_len=3, post_lead=-1, wait=2)     # pre: k-3..k-1 ; post: k+1..k+2


@dataclass
class LoopResult:
    history: list                     # [(t, Action)] including the final action
    final_action: Action
    t_final: int
    stages: list = field(default_factory=list)
    final_post: dict = field(default_factory=dict)
    fallback: bool = False


def _belief(pipe: str, area: float, t_est: int, cfg: Config) -> Incident:
    return Incident("BEL", "LEAK", t_est, {"pipe": pipe, "area": float(area), "cd": cfg.leak_cd})


def run_closed_loop(dg: Diagnoser, twin, rindex: RegionIndex, cfg: Config, p: ClosedLoopParams, k_alarm: int,
                    tel_fn, probe_sigma: np.ndarray, T_probe: float | None = None) -> LoopResult:
    """tel_fn(history) -> full-day telemetry of the TRUE plant under the actions taken so far (only data up to 'now' is used)."""
    lib, win, step = dg.lib, dg.win, cfg.step_s
    sens = dg.sensors
    n_pl = len(sens.pressure_nodes) + len(sens.tank_nodes)
    T_probe = p.T_probe if T_probe is None else T_probe
    k_dec = k_alarm + win.wait
    d = dg.diagnose(tel_fn([]).iloc[: k_dec + 1], k_alarm)
    t_est = max(0, (k_alarm - 2) * step)
    t = (k_dec + 1) * step
    tp = d.type_probs(lib, dg.T_type)
    leak_idx = {h.params["pipe"]: i for i, h in enumerate(lib.hyps) if h.kind == "LEAK"}
    post = d.leak_probs(lib, dg.T_loc)
    area = {pipe: max(float(d.amp[i]), 0.05) * lib.hyps[i].amp_unit for pipe, i in leak_idx.items()}
    history, closed, log = [], set(), []

    if tp["LEAK"] < p.min_leak_prob:                                   # not a leak: single-shot verified response
        from .incidents import Incident as _I
        from .diagnosis import belief_set
        beliefs = belief_set(dg, d, cfg)
        tab = build_table(twin, beliefs, t, cfg)
        i, _ = select_verified(tab, cfg)
        a = tab.pool[i] if i is not None else HOLD
        return LoopResult([(t, a)] if a.kind != "HOLD" else [], a, t, log, post, fallback=True)

    empty_gap = None
    for stage in range(1, p.max_stages + 1):
        top = sorted(post, key=post.get, reverse=True)[: p.top_m]
        if post[top[0]] >= p.stop_conf:
            break
        seeds = {n for pipe in top[:12] for n in rindex.ends[pipe]}
        q_leak = sum(post[x] * area[x] for x in top) * cfg.leak_cd * np.sqrt(2 * G_ACC * 40.0)     # m3/s, typical 40 m
        pv = np.array([v for v in post.values() if v > 0]); H_post = float(-(pv * np.log2(pv)).sum())      # bits still unknown
        remaining_h = max(0.0, (cfg.horizon_h * 3600 - t) / 3600.0)
        gap_h = p.stage_gap_steps * step / 3600.0
        best, best_score = None, (-np.inf if p.force else 0.0)
        for R in rindex.regions_around(seeds, p.max_boundary):
            m = sum(post.get(x, 0.0) for x in R.covers)
            if not (0.05 < m < 0.95) or set(R.boundary) == closed:
                continue
            hb = -(m * np.log2(m) + (1 - m) * np.log2(1 - m))                    # bits a probe can resolve
            score = p.value_scale * min(1.0, hb / max(H_post, 1e-9)) * q_leak * remaining_h - R.demand_m3s * gap_h
            if score > best_score:
                best, best_score = R, score
        if best is None:
            break
        act = Action("SET_PIPES", {"close": sorted(set(best.boundary) - closed), "open": sorted(closed - set(best.boundary))})
        history.append((t, act)); closed = set(best.boundary)
        k_a = t // step
        tel = tel_fn(history).iloc[: k_a + p.observe_steps + 1]
        if empty_gap is None or len(empty_gap) != len(tel):
            empty_gap = pd.DataFrame(index=tel.index)
        exps = dg.sync.expected_batch(tel, [(_belief(x, area[x], t_est, cfg), history) for x in top])
        chi = np.array([float(np.nansum((features(tel, e, empty_gap, sens, k_a, PROBE_WIN)[:n_pl] / probe_sigma[:n_pl]) ** 2)) for e in exps])
        w = np.array([post[x] for x in top]) * np.exp(-(chi - chi.min()) / (2.0 * T_probe))
        w = w / w.sum()
        log.append(dict(stage=stage, t=t, closed=len(best.boundary), covers_mass=round(sum(post.get(x, 0.0) for x in best.covers), 3),
                        outage_m3s=round(best.demand_m3s, 4), value_m3s_h=round(float(best_score), 4), top_before=top[0], top_after=top[int(np.argmax(w))],
                        post_after=round(float(w.max()), 3)))
        post = dict(zip(top, w.tolist()))
        t += p.stage_gap_steps * step

    # ---- final decision: verify candidates on the twin under the updated beliefs, with the history in place
    top5 = sorted(post, key=post.get, reverse=True)[:5]
    z = sum(post[x] for x in top5)
    beliefs = [(_belief(x, area[x], t_est, cfg), post[x] / z) for x in top5]
    base_open = sorted(closed)
    pool = candidate_pool(twin, beliefs, cfg)
    pool += [Action("SET_PIPES", {"close": [x], "open": [b for b in base_open if b != x]}) for x in top5[:3]]
    if len(top5) >= 2:
        pool.append(Action("SET_PIPES", {"close": top5[:2], "open": [b for b in base_open if b not in top5[:2]]}))
    if closed:
        pool.append(Action("SET_PIPES", {"open": base_open}))              # restore everything and live with the leak
    seen, uniq = set(), []
    for a in pool:
        if a.key() not in seen:
            seen.add(a.key()); uniq.append(a)
    from .policy import Table
    from .verifier import evaluate
    outs = [[evaluate(twin, b, a, t, cfg, history=history) for b, _ in beliefs] for a in uniq]
    tab = Table(uniq, np.array([w_ for _, w_ in beliefs]), outs)
    i, _ = select_verified(tab, cfg)
    final = uniq[i] if i is not None else HOLD
    full = history + ([(t, final)] if final.kind != "HOLD" else [])
    return LoopResult(full, final, t, log, post)
