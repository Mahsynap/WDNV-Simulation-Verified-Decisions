"""Shared experiment setup: plant, imperfect twin, signature library, sensors, synchronised twin, calibrated diagnoser."""
from __future__ import annotations
import numpy as np

from .config import Config
from .network import load_network
from .verifier import Model, make_twin, make_day, run_incident
from .incidents import generate_incidents
from .telemetry import choose_sensors, observe
from .sync import SyncTwin
from .diagnosis import build_library, NoiseModel, Diagnoser, WindowSpec, fit_temperature
from .evaluation import PipeGraph, true_hypothesis_index


class Stack:
    pass


def build_stack(net: str, cfg: Config, seed: int = 0, K: int = 25, noise: float = 0.5, twin_demand_sigma: float = 0.10,
                cal=(30, 10, 15), latency_steps: int = 3, lib_and_twin=None, n_normal: int = 30) -> Stack:
    from .closedloop import PROBE_WIN
    rng = np.random.default_rng(seed)
    st = Stack()
    st.cfg, st.rng, st.noise_sd, st.latency = cfg, rng, noise, latency_steps
    st.plant = Model.build(load_network(net, cfg), cfg)
    if lib_and_twin is None:
        st.twin = Model.build(make_twin(st.plant.wn, cfg, rng, twin_demand_sigma, 0.05, 0.10)[0], cfg)
        st.lib = build_library(st.twin, cfg, rng)
    else:
        st.twin, st.lib = lib_and_twin
    st.pg = PipeGraph(st.plant.wn)
    st.sens = choose_sensors(st.twin.base, K, rng)
    st.sync = SyncTwin(st.twin, st.sens, cfg)
    st.win = WindowSpec()
    normals = [observe(run_incident(make_day(st.plant.wn, rng), None, cfg)[1], st.sens, rng, p_sigma=noise) for _ in range(n_normal)]
    st.noise = NoiseModel.fit(st.sync, st.sens, normals, st.win, rng)
    st.probe_noise = NoiseModel.fit(st.sync, st.sens, normals, PROBE_WIN, rng)
    st.dg = Diagnoser(st.lib, st.sens, st.sync, st.noise, st.win)
    kinds = st.lib.kinds(); leak_ids = [i for i, k in enumerate(kinds) if k == "LEAK"]
    cal_set = []
    for inc in generate_incidents(st.plant.base, cfg, np.random.default_rng(seed + 1000), dict(zip(("LEAK", "PUMP_FAILURE", "DEMAND_SURGE"), cal))):
        _, res = run_incident(make_day(st.plant.wn, rng), inc, cfg)
        d = st.dg.diagnose(observe(res, st.sens, rng, p_sigma=noise), int(inc.t_start // cfg.step_s) + latency_steps)
        ti = true_hypothesis_index(st.lib, inc, st.pg)
        if ti >= 0:
            cal_set.append((inc, d, ti))
    st.dg.T_type = fit_temperature([d.chi2 for _, d, _ in cal_set], [t for _, _, t in cal_set])
    lc = [(d.chi2[:-1][leak_ids], leak_ids.index(t)) for inc, d, t in cal_set if inc.kind == "LEAK"]
    st.dg.T_loc = fit_temperature([c for c, _ in lc], [t for _, t in lc])
    return st
