"""Shared code for the three LLM-study phases: prepare packets -> (call a model elsewhere) -> evaluate proposals on the true plant."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
import numpy as np

from .actions import Action, HOLD, enumerate_candidates, validate_action
from .config import Config
from .diagnosis import belief_set
from .packet import build_packet
from .policy import Table, select_verified, naive_response
from .telemetry import observe
from .verifier import Model, make_day, run_incident, evaluate, select, model_with_incident


@dataclass
class Prepared:
    inc: object
    day: Model
    tel: object
    d: object
    beliefs: list
    packet: dict
    k_alarm: int
    t_act: int
    hold: object
    plant_cache: dict = field(default_factory=dict)
    twin_cache: dict = field(default_factory=dict)


def packet_hash(pk: dict) -> str:
    return hashlib.sha256(json.dumps(pk, sort_keys=True).encode()).hexdigest()[:16]


def prepare_incident(st, cfg: Config, seed: int, idx: int, inc, min_severity: float = 0.002):
    rng = np.random.default_rng([seed + 77, idx])
    wn_day = make_day(st.plant.wn, rng); day = Model.build(wn_day, cfg)
    seed_obs = int(rng.integers(1 << 30))
    k_alarm = int(inc.t_start // cfg.step_s) + st.latency
    k_dec = k_alarm + st.win.wait; t_act = (k_dec + 1) * cfg.step_s
    hold = evaluate(day, inc, HOLD, t_act, cfg)
    daily = st.plant.base.mean_demand_m3s * 86400
    if hold.status != "ok" or hold.objective < min_severity * daily:
        return None
    tel = observe(run_incident(wn_day, inc, cfg)[1], st.sens, np.random.default_rng(seed_obs), p_sigma=st.noise_sd)
    d = st.dg.diagnose(tel.iloc[: k_dec + 1], k_alarm)
    beliefs = belief_set(st.dg, d, cfg)
    pk = build_packet(tel, st.dg, d, st.twin, cfg, k_alarm)
    return Prepared(inc, day, tel, d, beliefs, pk, k_alarm, t_act, hold)


# ---- cached evaluations -------------------------------------------------------------------------------------------
def plant_eval(p: Prepared, cfg: Config, a: Action):
    k = a.key()
    if k not in p.plant_cache:
        p.plant_cache[k] = p.hold if a.kind == "HOLD" else evaluate(p.day, p.inc, a, p.t_act, cfg)
    return p.plant_cache[k]


def twin_row(p: Prepared, st, cfg: Config, a: Action):
    k = a.key()
    if k not in p.twin_cache:
        p.twin_cache[k] = [evaluate(st.twin, b, a, p.t_act, cfg) for b, _ in p.beliefs]
    return p.twin_cache[k]


def verified_choice(p: Prepared, st, cfg: Config, actions: list):
    uniq, seen = [], set()
    for a in [HOLD] + actions:
        if a.key() not in seen:
            seen.add(a.key()); uniq.append(a)
    tab = Table(uniq, np.array([w for _, w in p.beliefs]), [twin_row(p, st, cfg, a) for a in uniq])
    i, mode = select_verified(tab, cfg)
    return (uniq[i] if i is not None else HOLD), mode, tab, uniq


def search_actions(p: Prepared, st, cfg: Config) -> list:
    lp = p.d.leak_probs(st.lib, st.dg.T_loc)
    top = sorted(lp, key=lp.get, reverse=True)[:cfg.max_valve_ops]
    acts = [Action("CLOSE_PIPES", {"pipes": top[:k]}) for k in range(1, len(top) + 1)]
    for pump in st.twin.wn.pump_name_list:
        acts += [Action("SET_PUMP_STATUS", {"pump": pump, "status": "OPEN"}), Action("SET_PUMP_STATUS", {"pump": pump, "status": "CLOSED"})]
    return acts


def reference_outcomes(p: Prepared, st, cfg: Config) -> dict:
    pool = enumerate_candidates(model_with_incident(p.day, p.inc, cfg), p.inc, cfg)
    outs = [plant_eval(p, cfg, a) for a in pool]
    orc, _ = select(outs, p.hold, cfg); orc = orc if orc is not None else p.hold
    sa, sm, _, _ = verified_choice(p, st, cfg, search_actions(p, st, cfg))
    top = max(p.beliefs, key=lambda b: b[1])[0]
    nv = naive_response(top, st.twin, cfg)
    return {"oracle": orc, "search": plant_eval(p, cfg, sa), "search_mode": sm, "naive": plant_eval(p, cfg, nv), "hold": p.hold}


def _flags(o, hold):
    if o.status != "ok":
        return dict(cost=np.nan, unsafe=False, harmful=False, beneficial=False)
    return dict(cost=o.objective, unsafe=not o.hard_safe, harmful=o.objective > hold.objective * 1.05 + 5, beneficial=o.objective < hold.objective * 0.95 - 5)


def evaluate_response(p: Prepared, st, cfg: Config, parsed) -> dict:
    """Evaluate one parsed model response: proposal-level quality, the validator-filtered direct policy, and the twin-verified policy."""
    r = dict(parse_status=parsed.status, n_raw_actions=parsed.n_raw_actions, escalate=parsed.escalate)
    props, valid = [], []
    for a in parsed.actions[:3]:
        errs = validate_action(st.twin.wn, a, cfg) if a.kind not in ("HOLD", "OPERATOR_REVIEW") else ([] if a.kind == "HOLD" else ["review"])
        unknown_id = any(("unknown pipe" in e or "unknown pump" in e or "not an operable" in e) for e in errs)
        bad_kind = any("unsupported action kind" in e for e in errs)
        rec = dict(kind=a.kind, valid=not errs, unknown_id=unknown_id, bad_kind=bad_kind, review=(a.kind == "OPERATOR_REVIEW"))
        if not errs:
            rec.update(_flags(plant_eval(p, cfg, a), p.hold))
            if a.kind != "HOLD":
                valid.append(a)
        props.append(rec)
    r["proposals"] = props
    r["n_proposed"] = len(props); r["n_invalid"] = sum((not x["valid"]) and not x["review"] for x in props)
    r["first_valid_key"] = (valid[0].key() if valid else "HOLD")
    # direct (validator-filtered): first valid executable action; escalation or nothing valid -> deferred (HOLD)
    direct = valid[0] if (valid and not parsed.escalate) else HOLD
    r["direct_deferred"] = (not valid) or parsed.escalate
    od = plant_eval(p, cfg, direct); f = _flags(od, p.hold)
    r.update(direct_cost=f["cost"], direct_unsafe=f["unsafe"], direct_harmful=f["harmful"])
    # verified: twin chooses among the model's valid proposals (+ HOLD) under the belief set
    ch, mode, tab, uniq = verified_choice(p, st, cfg, valid)
    ov = plant_eval(p, cfg, ch); f = _flags(ov, p.hold)
    r.update(verified_cost=f["cost"], verified_unsafe=f["unsafe"], verified_harmful=f["harmful"], verified_mode=mode, verified_is_hold=(ch.kind == "HOLD"))
    # twin verdict vs plant verdict for each valid proposal
    conf = []
    for a in valid:
        i = [k for k, x in enumerate(uniq) if x.key() == a.key()][0]
        conf.append((bool(tab.safe(i, 0.10)), bool(plant_eval(p, cfg, a).hard_safe)))
    r["twin_vs_plant"] = conf
    return r
