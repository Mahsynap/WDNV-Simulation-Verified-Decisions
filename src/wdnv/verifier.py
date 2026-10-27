from __future__ import annotations
import copy
from dataclasses import dataclass, field
import numpy as np

from .config import Config
from .actions import Action, HOLD, validate_action, apply_action
from .incidents import Incident, apply_incident
from .metrics import compute_metrics, hard_violations, severity
from .network import Baseline, compute_baseline, run_epanet


@dataclass
class Model:
    """A hydraulic model (the plant, or a twin of it) together with its own no-incident baseline."""
    wn: object
    base: Baseline

    @staticmethod
    def build(wn, cfg: Config) -> "Model":
        return Model(wn, compute_baseline(wn, cfg))


@dataclass
class Outcome:
    action: Action
    status: str                       # ok | invalid | sim_failed
    metrics: dict = field(default_factory=dict)
    violations: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    deferred: bool = False            # True for OPERATOR_REVIEW (evaluated as HOLD)
    results: object = None            # raw EPANET results (only if keep_results)

    @property
    def hard_safe(self) -> bool:
        return self.status == "ok" and not self.violations

    @property
    def objective(self) -> float:
        return self.metrics.get("objective_m3", float("inf")) if self.status == "ok" else float("inf")


def model_with_incident(model: Model, inc: Incident | None, cfg: Config):
    wn = copy.deepcopy(model.wn)
    if inc is not None:
        apply_incident(wn, inc, cfg)
    return wn


def evaluate(model: Model, inc: Incident | None, action: Action, t_action: int, cfg: Config,
             keep_results: bool = False, history: list | None = None) -> Outcome:
    """Simulate `inc` with the earlier actions `history` = [(t, Action), ...] followed by `action` at `t_action`.
    Constraints are checked from the first action time on; the operation count covers the whole sequence."""
    wn = model_with_incident(model, inc, cfg)
    deferred = action.kind == "OPERATOR_REVIEW"
    eff = HOLD if deferred else action
    history = list(history or [])
    for _, ha in history:
        errs = validate_action(wn, ha, cfg) if ha.kind not in ("HOLD", "OPERATOR_REVIEW") else []
        if errs:
            return Outcome(action, "invalid", errors=["history: " + e for e in errs])
    if action.kind not in ("HOLD", "OPERATOR_REVIEW"):
        errs = validate_action(wn, action, cfg)
        if errs:
            return Outcome(action, "invalid", errors=errs)
    for ht, ha in history:
        apply_action(wn, ha, ht, cfg)
    apply_action(wn, eff, t_action, cfg)
    t_first = min([t for t, _ in history] + [t_action])
    n_ops = sum(a.n_ops() for _, a in history) + eff.n_ops()
    try:
        res = run_epanet(wn)
    except Exception as e:                                   # solver failure is NOT a safety verdict
        return Outcome(action, "sim_failed", errors=[f"{type(e).__name__}: {str(e)[:120]}"])
    t_ref = inc.t_start if inc is not None else 0
    m = compute_metrics(res, wn, model.base, inc, n_ops, cfg, t_ref, t_first)
    return Outcome(action, "ok", metrics=m, violations=hard_violations(m, cfg), deferred=deferred,
                   results=res if keep_results else None)


def select(outcomes: list, hold: Outcome, cfg: Config):
    """Return (chosen Outcome or None, mode).
    mode: 'safe_best' | 'damage_limiting' | 'review'."""
    ok = [o for o in outcomes if o.status == "ok" and o.action.executable()]
    safe = sorted([o for o in ok if o.hard_safe], key=lambda o: o.objective)
    if safe:
        return safe[0], "safe_best"
    # no hard-safe action exists: accept only actions that do not worsen any violation vs HOLD and improve the objective
    hs = severity(hold.metrics, cfg) if hold.status == "ok" else (0, 0, 0)
    ok_dl = [o for o in ok if all(a <= b + 1e-9 for a, b in zip(severity(o.metrics, cfg), hs))
             and o.objective < hold.objective]
    if ok_dl:
        return sorted(ok_dl, key=lambda o: o.objective)[0], "damage_limiting"
    return None, "review"


def make_twin(plant_wn, cfg: Config, rng, demand_sigma=0.10, node_sigma=0.05, rough_sigma=0.10):
    """Imperfect copy of the plant: global + per-node demand error, pipe-roughness error."""
    wn = copy.deepcopy(plant_wn)
    g = float(np.exp(rng.normal(0.0, demand_sigma)))
    wn.options.hydraulic.demand_multiplier = wn.options.hydraulic.demand_multiplier * g
    for jn in wn.junction_name_list:
        f = float(np.exp(rng.normal(0.0, node_sigma)))
        for ts in wn.get_node(jn).demand_timeseries_list:
            ts.base_value = ts.base_value * f
    for _, p in wn.pipes():
        p.roughness = float(p.roughness * np.exp(rng.normal(0.0, rough_sigma)))
    return wn, {"demand_global_factor": g}


def make_day(plant_wn, rng, global_sigma: float = 0.04, node_sigma: float = 0.05):
    """One 'real' day of the plant: demand differs from the nominal model (day-to-day variability).
    Roughness and topology are unchanged. Used so that detectors cannot memorise a single normal day."""
    wn = copy.deepcopy(plant_wn)
    g = float(np.exp(rng.normal(0.0, global_sigma)))
    wn.options.hydraulic.demand_multiplier = wn.options.hydraulic.demand_multiplier * g
    for jn in wn.junction_name_list:
        f = float(np.exp(rng.normal(0.0, node_sigma)))
        for ts in wn.get_node(jn).demand_timeseries_list:
            ts.base_value = ts.base_value * f
    return wn


def run_incident(wn, inc: Incident | None, cfg: Config, horizon_s: int | None = None, history: list | None = None):
    """Simulate `wn` (+ optional incident and operator actions [(t, Action)]) and return (wn_run, results)."""
    w = copy.deepcopy(wn)
    if inc is not None:
        apply_incident(w, inc, cfg)
    for ht, ha in (history or []):
        apply_action(w, ha, ht, cfg)
    if horizon_s is not None:
        w.options.time.duration = int(horizon_s)
    return w, run_epanet(w)
