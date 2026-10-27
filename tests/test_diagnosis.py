import numpy as np
import pytest
from wdnv.actions import Action, HOLD
from wdnv.diagnosis import (build_library, NoiseModel, Diagnoser, WindowSpec, belief_set, fit_temperature, channel_names)
from wdnv.evaluation import ece, credible_set, PipeGraph, true_hypothesis_index
from wdnv.incidents import Incident, apply_incident
from wdnv.policy import Table, select_verified, naive_response, candidate_pool
from wdnv.sync import SyncTwin
from wdnv.telemetry import choose_sensors, observe
from wdnv.verifier import Model, make_day, run_incident, Outcome

QUIET = dict(p_sigma=0.0, q_rel=0.0, level_sigma=0.0, dropout=0.0)


@pytest.fixture(scope="module")
def stack(net3, cfg):
    """Perfect twin (twin == plant), 25 pressure sensors, noiseless telemetry: isolates logic from uncertainty."""
    rng = np.random.default_rng(0)
    lib = build_library(net3, cfg, rng)
    sens = choose_sensors(net3.base, 25, rng)
    sync = SyncTwin(net3, sens, cfg)
    win = WindowSpec()
    normals = []
    for _ in range(12):
        _, res = run_incident(net3.wn, None, cfg)
        normals.append(observe(res, sens, rng, **QUIET))
    noise = NoiseModel.fit(sync, sens, normals, win, rng)
    return dict(lib=lib, sens=sens, sync=sync, win=win, dg=Diagnoser(lib, sens, sync, noise, win), rng=rng)


def tel(net3, cfg, inc, stack):
    _, res = run_incident(net3.wn, inc, cfg)
    return observe(res, stack["sens"], stack["rng"], **QUIET)


def test_sync_twin_reproduces_the_plant_when_there_is_no_mismatch(net3, cfg, stack):
    t = tel(net3, cfg, None, stack)
    e = stack["sync"].expected(t)
    pc = [c for c in t.columns if c.startswith("P:")]
    err = np.abs(t[pc].values - e[pc].values)
    assert np.percentile(err, 95) < 0.6 and err.max() < 3.0          # one tank-threshold timing event may reach ~2 m
    assert np.abs(stack["sync"].status_gap(t).values).max() == 0


def test_pump_failure_shows_up_as_persistent_status_gap(net3, cfg, stack):
    inc = Incident("PUM000", "PUMP_FAILURE", 6 * 3600, {"pump": "10"})
    g = stack["sync"].status_gap(tel(net3, cfg, inc, stack))
    k0 = inc.t_start // cfg.step_s
    assert (g["10"].values[k0 + 1: k0 + 8] == -1).all()
    assert (g["335"].values == 0).all()
    assert stack["sync"].status_alarm(tel(net3, cfg, inc, stack), persist=3) is not None


def test_library_layout(stack, net3):
    lib, sens = stack["lib"], stack["sens"]
    M = lib.matrix(sens)
    assert M.shape == (len(lib.hyps), len(channel_names(sens)))
    kinds = lib.kinds()
    assert {"LEAK", "PUMP_FAILURE", "DEMAND_SURGE"} <= set(kinds)
    leakP = lib.P[kinds == "LEAK"]
    assert (leakP.mean(axis=1) < 0).mean() > 0.9               # leaks lower pressure on average


def test_noiseless_perfect_twin_recovers_the_truth(net3, cfg, stack):
    dg, lib, win = stack["dg"], stack["lib"], stack["win"]
    ok_leak = 0
    for pipe, t0 in (("109", 3 * 3600 + 900), ("207", 7 * 3600), ("317", 5 * 3600)):
        inc = Incident("L", "LEAK", t0, {"pipe": pipe, "area": 0.006, "cd": cfg.leak_cd})
        d = dg.diagnose(tel(net3, cfg, inc, stack), t0 // cfg.step_s + 3)
        tp = d.type_probs(lib, 1.0)
        lp = d.leak_probs(lib, 1.0)
        top5 = sorted(lp, key=lp.get, reverse=True)[:5]
        ok_leak += (tp["LEAK"] > 0.4) and (pipe in top5)
    assert ok_leak >= 2
    inc = Incident("P", "PUMP_FAILURE", 6 * 3600, {"pump": "10"})
    d = dg.diagnose(tel(net3, cfg, inc, stack), 6 * 4 + 3)
    assert d.type_probs(lib, 1.0)["PUMP_FAILURE"] > 0.9


def test_beliefs_are_valid_incidents_with_normalised_weights(net3, cfg, stack):
    dg, lib = stack["dg"], stack["lib"]
    inc = Incident("L", "LEAK", 4 * 3600, {"pipe": "207", "area": 0.006, "cd": cfg.leak_cd})
    d = dg.diagnose(tel(net3, cfg, inc, stack), 4 * 4 + 3)
    bs = belief_set(dg, d, cfg)
    assert bs and abs(sum(w for _, w in bs) - 1.0) < 1e-9
    import copy
    for b, _ in bs:
        apply_incident(copy.deepcopy(net3.wn), b, cfg)             # must be applicable to a model


def test_temperature_fit_and_calibration_utils():
    rng = np.random.default_rng(0)
    chi = [np.array([0.0, 30.0, 30.0]) for _ in range(50)]
    assert fit_temperature(chi, [0] * 50) < 1.0                    # confident and always right -> sharp
    assert fit_temperature(chi, [1] * 50) > 10.0                   # confident and always wrong -> flattened
    conf = rng.random(4000); corr = rng.random(4000) < conf
    assert ece(conf, corr) < 0.05
    assert ece(np.full(1000, 0.9), np.zeros(1000)) > 0.85
    p = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}
    assert credible_set(p, 0.8) == ["a", "b"] and credible_set(p, 0.95) == ["a", "b", "c"]


def test_pipe_graph_distances(net3):
    pg = PipeGraph(net3.wn)
    assert pg.hop("109", "109") == 0
    assert pg.hop("109", "207") == pg.hop("207", "109") >= 0


def _tab(weights, objs, safe):
    pool = [HOLD] + [Action("SET_PUMP_STATUS", {"pump": f"p{i}", "status": "OPEN"}) for i in range(len(objs) - 1)]
    outs = []
    for i, row in enumerate(objs):
        outs.append([Outcome(pool[i], "ok", metrics={"objective_m3": o, "min_pressure_eligible_m": 30.0 if s else 2.0,
                                                    "envelope_excess_m": 0.0, "tanks_emptied": 0},
                             violations=[] if s else ["min_pressure"]) for o, s in zip(row, safe[i])])
    return Table(pool, np.array(weights), outs)


def test_robust_selection_respects_belief_weights(cfg):
    # belief 1 (70%): action A is great, action B is mediocre. belief 2 (30%): A is catastrophic and unsafe.
    t = _tab([0.7, 0.3], [[100, 100], [10, 100], [60, 60]], [[1, 1], [1, 0], [1, 1]])
    i, mode = select_verified(t, cfg, p_min=0.1)
    assert t.pool[i].params["pump"] == "p1" and mode == "safe_best"          # B: safe under both beliefs (A cheaper but unsafe under belief 2)
    i, mode = select_verified(t, cfg, p_min=0.5)                              # a 30% belief no longer vetoes A
    assert t.pool[i].params["pump"] == "p0"
    assert t.regret(1) < t.regret(2)                                          # A has lower expected regret than B


def test_naive_response_follows_top_belief(net3, cfg):
    leak = Incident("B", "LEAK", 0, {"pipe": "207", "area": 0.001})
    assert naive_response(leak, net3, cfg).params["pipes"] == ["207"]
    pf = Incident("B", "PUMP_FAILURE", 0, {"pump": "10"})
    assert naive_response(pf, net3, cfg).params == {"pump": "335", "status": "OPEN"}
    assert naive_response(Incident("B", "DEMAND_SURGE", 0, {}), net3, cfg).kind == "HOLD"


def test_uncontrolled_always_on_pump_is_monitored_too(ky6, cfg):
    rng = np.random.default_rng(1)
    sens = choose_sensors(ky6.base, 15, rng)
    sync = SyncTwin(ky6, sens, cfg)
    uncontrolled = [p for p in ky6.wn.pump_name_list if p not in sync.rules]
    assert uncontrolled, "KY6 has an always-on pump without controls"
    p = uncontrolled[0]
    inc = Incident("PUM", "PUMP_FAILURE", 6 * 3600, {"pump": p})
    _, res = run_incident(ky6.wn, inc, cfg)
    g = sync.status_gap(observe(res, sens, rng, **QUIET))
    assert (g[p].values[6 * 4 + 1: 6 * 4 + 8] == -1).all()


def test_missing_pump_hypothesis_is_reported_not_crashed(net3, cfg, stack):
    pg = PipeGraph(net3.wn)
    ghost = Incident("G", "PUMP_FAILURE", 6 * 3600, {"pump": "ghost"})
    assert true_hypothesis_index(stack["lib"], ghost, pg) == -1
