import copy
import numpy as np
import wntr
from wdnv.config import Config
from wdnv.network import load_network, run_epanet, requested_demand, resolve_path, drop_controls_for
from wdnv.incidents import Incident, apply_incident, leak_names
from wdnv.verifier import model_with_incident


def test_timing_standardization_preserves_hourly_behaviour(cfg):
    ref = wntr.network.WaterNetworkModel(resolve_path("Net3"))
    ref.options.time.duration = 24 * 3600
    ref.options.hydraulic.demand_model = "PDD"
    ref.options.hydraulic.minimum_pressure, ref.options.hydraulic.required_pressure = 7.0, 20.0
    pr = run_epanet(ref).node["pressure"][ref.junction_name_list]
    new = load_network("Net3", cfg)
    pn = run_epanet(new).node["pressure"][new.junction_name_list]
    pn_h = pn.loc[pn.index % 3600 == 0]
    assert np.abs(pr.values - pn_h.values[: len(pr)]).max() < 0.3


def test_baseline_demand_accounting(net3):
    b = net3.base
    req = requested_demand(net3.wn, b.times)
    dl = b.results.node["demand"][b.junctions]
    ratio = dl[b.eligible].values.sum() / req[b.eligible].values.sum()
    assert abs(ratio - 1.0) < 1e-3


def test_leak_orifice_matches_analytic_and_wntr_leak_model(net3, cfg, leak109):
    wn = model_with_incident(net3, leak109, cfg)
    res = run_epanet(wn)
    nm = leak_names(leak109, cfg)
    q = res.link["flowrate"][nm["link"]]
    p = res.node["pressure"][nm["node"]]
    t0 = leak109.t_start
    assert q[q.index < t0].abs().max() < 1e-9                                     # no leak before onset
    on = q.index >= t0
    analytic = cfg.leak_cd * leak109.params["area"] * np.sqrt(2 * 9.81 * p[on].clip(lower=0))
    assert np.max(np.abs(q[on].values - analytic.values) / analytic.values) < 0.01   # <1% orifice error

    # independent check: WNTR's own leak model on the same split network
    ref = load_network("Net3", cfg)
    wntr.morph.split_pipe(ref, nm["pipe_a"], nm["pipe_b"], nm["node"], return_copy=False)
    ref.get_node(nm["node"]).add_leak(ref, area=leak109.params["area"], discharge_coeff=cfg.leak_cd, start_time=t0)
    r2 = wntr.sim.WNTRSimulator(ref).run_sim()
    J = net3.base.junctions
    d = np.abs(res.node["pressure"][J].values - r2.node["pressure"][J].values).max()
    assert d < 0.1


def test_nothing_changes_before_onset_for_all_incident_kinds(net3, cfg, leak109):
    J = net3.base.junctions
    incs = [leak109,
            Incident("PUM000", "PUMP_FAILURE", 5 * 3600, {"pump": "10"}),
            Incident("SUR000", "DEMAND_SURGE", 6 * 3600, {"nodes": J[:10], "factor": 2.0, "duration": 7200})]
    for inc in incs:
        res = run_epanet(model_with_incident(net3, inc, cfg))
        pre = net3.base.times < inc.t_start
        a = res.node["pressure"][J].values[: pre.sum()]
        b = net3.base.pressure[J].values[pre]
        assert np.allclose(a, b, atol=1e-6), inc.kind


def test_pump_failure_is_permanent_and_leaves_earlier_schedule_intact(net3, cfg):
    inc = Incident("PUM000", "PUMP_FAILURE", 5 * 3600, {"pump": "10"})       # Net3 pump 10 has a daily schedule
    res = run_epanet(model_with_incident(net3, inc, cfg))
    q = res.link["flowrate"]["10"]
    assert q[q.index >= inc.t_start].abs().max() < 1e-9                      # stays off, later 'open' controls ignored
    base_q = net3.base.results.link["flowrate"]["10"]
    assert (base_q[base_q.index >= 25 * 0 + 5 * 3600 + 900].abs() > 1e-6).any()   # ...although baseline would run it again


def test_demand_surge_scales_requested_demand_only_inside_window(net3, cfg):
    J = net3.base.junctions
    nodes = [j for j in J if net3.wn.get_node(j).demand_timeseries_list[0].base_value > 0][:8]
    inc = Incident("SUR000", "DEMAND_SURGE", 6 * 3600, {"nodes": nodes, "factor": 2.0, "duration": 3 * 3600})
    wn = model_with_incident(net3, inc, cfg)
    t = net3.base.times
    r0, r1 = requested_demand(net3.wn, t)[nodes].sum(axis=1), requested_demand(wn, t)[nodes].sum(axis=1)
    inside = (t >= 6 * 3600) & (t < 9 * 3600)
    assert np.allclose(r1[inside] / r0[inside], 2.0, rtol=1e-6)
    assert np.allclose(r1[~inside], r0[~inside], rtol=1e-9)
