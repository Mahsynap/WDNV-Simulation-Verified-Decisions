import numpy as np
import pytest
from wdnv.actions import Action, HOLD, REVIEW, validate_action, enumerate_candidates
from wdnv.incidents import Incident, leak_names
from wdnv.network import run_epanet
import wdnv.verifier as V
from wdnv.verifier import evaluate, select, Outcome, model_with_incident, make_twin, Model
from wdnv.network import compute_baseline
from wdnv.telemetry import choose_sensors, observe


def test_action_validation_rejects_hallucinated_and_unsafe_commands(net3, cfg, leak109):
    wn = model_with_incident(net3, leak109, cfg)
    good = Action("CLOSE_PIPES", {"pipes": ["109"]})
    assert validate_action(wn, good, cfg) == []
    bad = [Action("OPEN_ALL_VALVES"),
           Action("CLOSE_PIPES", {"pipes": ["NOPE"]}),
           Action("CLOSE_PIPES", {"pipes": []}),
           Action("CLOSE_PIPES", {"pipes": ["109", "109"]}),
           Action("CLOSE_PIPES", {"pipes": [f"{i}" for i in list(wn.pipe_name_list)[:cfg.max_valve_ops + 1]]}),
           Action("CLOSE_PIPES", {"pipes": [leak_names(leak109, cfg)["link"]]}),
           Action("CLOSE_PIPES", {"pipes": ["109__b"]}),                      # internal half-pipe names are not operable
           Action("SET_PUMP_STATUS", {"pump": "10", "status": "HALF"}),
           Action("SET_PUMP_STATUS", {"pump": "ghost", "status": "OPEN"}),
           Action("SET_PUMP_SPEED", {"pump": "10", "speed": 2.5}),
           Action.from_dict("close everything")]
    for a in bad:
        assert validate_action(wn, a, cfg), a.to_dict()


def test_isolation_stops_leak_through_hydraulics_not_by_assumption(net3, cfg, leak109):
    ta = leak109.t_start + 3600
    hold = evaluate(net3, leak109, HOLD, ta, cfg)
    iso = evaluate(net3, leak109, Action("CLOSE_PIPES", {"pipes": ["109"]}), ta, cfg, keep_results=True)
    wrong = evaluate(net3, leak109, Action("CLOSE_PIPES", {"pipes": ["117"]}), ta, cfg)         # isolating the wrong pipe
    assert hold.status == iso.status == wrong.status == "ok"
    q = iso.results.link["flowrate"][leak_names(leak109, cfg)["link"]]
    assert q[q.index >= ta].abs().max() < 1e-6                       # leak flow ~0 after isolation
    assert iso.metrics["leak_m3"] < 0.3 * hold.metrics["leak_m3"]
    assert iso.objective < hold.objective
    assert wrong.metrics["leak_m3"] > 5 * iso.metrics["leak_m3"]      # isolating the wrong pipe does not stop the leak


def test_exact_isolation_is_the_oracle_choice_for_a_burst(net3, cfg, leak109):
    ta = leak109.t_start + 3600
    pool = enumerate_candidates(model_with_incident(net3, leak109, cfg), leak109, cfg, np.random.default_rng(0))
    outs = [evaluate(net3, leak109, a, ta, cfg) for a in pool]
    hold = outs[0]
    best, mode = select(outs, hold, cfg)
    assert best is not None
    assert best.action.kind == "CLOSE_PIPES" and "109" in best.action.params["pipes"]
    assert best.objective < 0.5 * hold.objective


def test_solver_failure_is_reported_separately_from_safety(net3, cfg, leak109, monkeypatch):
    def boom(wn): raise RuntimeError("solver exploded")
    monkeypatch.setattr(V, "run_epanet", boom)
    o = evaluate(net3, leak109, HOLD, leak109.t_start + 3600, cfg)
    assert o.status == "sim_failed" and not o.hard_safe and o.violations == []
    best, mode = select([o], o, cfg)
    assert best is None and mode == "review"


def _fake(kind, obj, viol=(), minp=30.0, rise=0.0, emptied=0, ok=True, params=None):
    m = {"objective_m3": obj, "min_pressure_eligible_m": minp, "envelope_excess_m": rise, "tanks_emptied": emptied}
    return Outcome(Action(kind, params or {}), "ok" if ok else "invalid", metrics=m, violations=list(viol))


def test_selection_modes(cfg):
    hold = _fake("HOLD", 1000, viol=["min_pressure"], minp=4.0)
    safe = _fake("SET_PUMP_STATUS", 400)
    assert select([hold, safe], hold, cfg)[1] == "safe_best"
    better_but_unsafe = _fake("CLOSE_PIPES", 300, viol=["min_pressure"], minp=6.0)
    chosen, mode = select([hold, better_but_unsafe], hold, cfg)
    assert mode == "damage_limiting" and chosen is better_but_unsafe
    worse = _fake("CLOSE_PIPES", 300, viol=["min_pressure"], minp=2.0)             # improves objective but deepens violation
    assert select([hold, worse], hold, cfg)[1] == "review"
    assert select([hold], hold, cfg)[1] == "review"


def test_twin_is_imperfect_but_reasonable(net3, cfg):
    twin_wn, info = make_twin(net3.wn, cfg, np.random.default_rng(3))
    tw = Model.build(twin_wn, cfg)
    d = tw.base.pressure[net3.base.junctions].values - net3.base.pressure[net3.base.junctions].values
    rms = float(np.sqrt((d ** 2).mean()))
    assert 0.01 < rms < 10.0


def test_telemetry_shapes_noise_and_dropout(net3, cfg):
    rng = np.random.default_rng(0)
    s = choose_sensors(net3.base, 10, rng)
    assert len(set(s.pressure_nodes)) == 10
    df = observe(net3.base.results, s, rng, p_sigma=0.5, dropout=0.05)
    col = f"P:{s.pressure_nodes[0]}"
    truth = net3.base.results.node["pressure"][s.pressure_nodes[0]]
    resid = (df[col] - truth).dropna()
    assert 0.3 < resid.std() < 0.7
    assert 0.0 < df.isna().values.mean() < 0.15


def test_ky6_loads_and_handles_a_burst(ky6, cfg):
    b = ky6.base
    assert len(b.eligible) > 0.5 * len(b.junctions)
    from wdnv.incidents import sample_leak
    inc = sample_leak(b, cfg, np.random.default_rng(5), 0)
    ta = inc.t_start + 3600
    hold = evaluate(ky6, inc, HOLD, ta, cfg)
    iso = evaluate(ky6, inc, Action("CLOSE_PIPES", {"pipes": [inc.params["pipe"]]}), ta, cfg)
    assert hold.status == iso.status == "ok"
    assert iso.metrics["leak_m3"] < hold.metrics["leak_m3"]


def test_candidate_actions_do_not_alter_the_trajectory_before_they_are_applied(net3, cfg, leak109):
    ta = leak109.t_start + 3600
    hold = evaluate(net3, leak109, HOLD, ta, cfg, keep_results=True)
    pool = enumerate_candidates(model_with_incident(net3, leak109, cfg), leak109, cfg, np.random.default_rng(0))
    J = net3.base.junctions
    pre = hold.results.node["pressure"].index.values < ta
    ref = hold.results.node["pressure"][J].values[pre]
    n_checked = 0
    for a in pool:
        o = evaluate(net3, leak109, a, ta, cfg, keep_results=True)
        if o.status != "ok":
            continue
        assert np.allclose(o.results.node["pressure"][J].values[pre], ref, atol=1e-6), a.to_dict()
        n_checked += 1
    assert n_checked >= 8


def test_deliberately_cut_off_customers_are_an_outage_not_a_pressure_violation(net3, cfg):
    """Isolating a burst on a dead-end feeder cuts customers off (charged as unmet demand) without flagging min_pressure."""
    from wdnv.incidents import Incident
    pipe = "149"
    l = net3.wn.get_link(pipe)
    p_est = float(net3.base.pressure[[l.start_node_name, l.end_node_name]].mean().mean())
    area = 0.08 * net3.base.mean_demand_m3s / (cfg.leak_cd * np.sqrt(2 * 9.81 * p_est))
    inc = Incident("X", "LEAK", 4 * 3600, {"pipe": pipe, "area": float(area), "cd": cfg.leak_cd})
    ta = inc.t_start + 6300
    hold = evaluate(net3, inc, HOLD, ta, cfg)
    iso = evaluate(net3, inc, Action("CLOSE_PIPES", {"pipes": [pipe]}), ta, cfg)
    assert iso.metrics["n_nodes_cut_off"] >= 1 and iso.metrics["unmet_m3"] > 100
    assert "min_pressure" not in iso.violations
    assert iso.objective < 0.5 * hold.objective
    best, mode = select([hold, iso], hold, cfg)
    assert best.action.kind == "CLOSE_PIPES"
