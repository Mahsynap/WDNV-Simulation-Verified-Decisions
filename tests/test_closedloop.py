import numpy as np
import pytest
from wdnv.actions import Action, HOLD, validate_action
from wdnv.incidents import Incident, leak_names
from wdnv.regions import RegionIndex
from wdnv.diagnosis import _lib_pipes
from wdnv.verifier import evaluate, model_with_incident, run_incident
from wdnv.network import run_epanet


def test_set_pipes_validation(net3, cfg, leak109):
    wn = model_with_incident(net3, leak109, cfg)
    assert validate_action(wn, Action("SET_PIPES", {"close": ["109"], "open": ["117"]}), cfg) == []
    for bad in (Action("SET_PIPES", {}), Action("SET_PIPES", {"close": ["109"], "open": ["109"]}),
                Action("SET_PIPES", {"close": ["nope"]}), Action("SET_PIPES", {"close": ["109__b"]}),
                Action("SET_PIPES", {"close": list(wn.pipe_name_list)[: cfg.max_probe_ops + 1]})):
        assert validate_action(wn, bad, cfg), bad.to_dict()


def test_regions_are_cut_off_from_all_sources_and_cover_their_pipes(net3, cfg):
    ri = RegionIndex(net3.wn, _lib_pipes(net3.wn))
    seeds = list(net3.base.junctions)[:12]
    regs = ri.regions_around(seeds, 12)
    assert regs
    for R in regs[:15]:
        assert not (R.nodes & ri.special)
        assert 1 <= len(R.boundary) <= 12
        assert all(p in R.covers for p in R.covers if ri.ends[p][0] in R.nodes or ri.ends[p][1] in R.nodes)
        # closing the boundary really disconnects the region (graph check)
        G = ri.G.copy()
        for p in R.boundary:
            a, b = ri.ends[p]
            if G.has_edge(a, b):
                G.remove_edge(a, b)
        outside = next(iter(set(G.nodes) - R.nodes - ri.special - {n for n in G.nodes if G.degree(n) == 0}))
        import networkx as nx
        assert not any(nx.has_path(G, n, outside) for n in R.nodes)


def test_closing_a_region_around_the_leak_stops_it_and_wrong_region_does_not(net3, cfg, leak109):
    ri = RegionIndex(net3.wn, _lib_pipes(net3.wn))
    seeds = list(ri.ends["109"])
    regs = [R for R in ri.regions_around(seeds, 12) if "109" in R.covers]
    assert regs
    R = min(regs, key=lambda r: len(r.boundary))
    ta = leak109.t_start + 3600
    hold = evaluate(net3, leak109, HOLD, ta, cfg)
    inside = evaluate(net3, leak109, Action("SET_PIPES", {"close": list(R.boundary)}), ta, cfg)
    far = [R2 for R2 in ri.regions_around(list(net3.base.junctions)[40:60], 12) if "109" not in R2.covers]
    outside = evaluate(net3, leak109, Action("SET_PIPES", {"close": list(far[0].boundary)}), ta, cfg)
    assert inside.metrics["leak_m3"] < 0.4 * hold.metrics["leak_m3"]
    assert outside.metrics["leak_m3"] > 0.8 * hold.metrics["leak_m3"]


def test_history_changes_the_outcome_and_counts_all_operations(net3, cfg, leak109):
    ta = leak109.t_start + 3600
    first = (ta, Action("SET_PIPES", {"close": ["109"]}))
    undo = Action("SET_PIPES", {"open": ["109"]})
    with_undo = evaluate(net3, leak109, undo, ta + 4 * cfg.step_s, cfg, history=[first])
    without = evaluate(net3, leak109, HOLD, ta + 4 * cfg.step_s, cfg, history=[first])
    assert with_undo.metrics["leak_m3"] > without.metrics["leak_m3"]          # re-opening the burst pipe lets it leak again
    assert with_undo.metrics["ops"] == 2 and without.metrics["ops"] == 1


def test_probe_likelihood_separates_true_from_wrong_hypothesis(net3, cfg, leak109):
    """Noise-free, perfect twin: after a probe that cuts off the true pipe, the true hypothesis explains the telemetry better."""
    from wdnv.diagnosis import features, WindowSpec
    from wdnv.closedloop import PROBE_WIN
    from wdnv.sync import SyncTwin
    from wdnv.telemetry import choose_sensors, observe
    import pandas as pd
    rng = np.random.default_rng(0)
    sens = choose_sensors(net3.base, 25, rng)
    sync = SyncTwin(net3, sens, cfg)
    ri = RegionIndex(net3.wn, _lib_pipes(net3.wn))
    R = min([r for r in ri.regions_around(list(ri.ends["109"]), 12) if "109" in r.covers], key=lambda r: len(r.boundary))
    t = leak109.t_start + 3600
    hist = [(t, Action("SET_PIPES", {"close": list(R.boundary)}))]
    _, res = run_incident(net3.wn, leak109, cfg, history=hist)
    k_a = t // cfg.step_s
    tel = observe(res, sens, rng, p_sigma=0.0, q_rel=0.0, level_sigma=0.0, dropout=0.0).iloc[: k_a + 3]
    wrong = next(p for p in _lib_pipes(net3.wn) if p not in R.covers and p != "109")
    def chi(pipe):
        b = Incident("BEL", "LEAK", leak109.t_start, {"pipe": pipe, "area": leak109.params["area"], "cd": cfg.leak_cd})
        e = sync.expected_batch(tel, [(b, hist)])[0]
        d = features(tel, e, pd.DataFrame(index=tel.index), sens, k_a, PROBE_WIN)[: len(sens.pressure_nodes) + len(sens.tank_nodes)]
        return float(np.nansum(d ** 2))
    assert chi("109") < 0.2 * chi(wrong)
