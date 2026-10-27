from __future__ import annotations
from dataclasses import dataclass
import networkx as nx
import numpy as np
import pandas as pd

from .network import Baseline


@dataclass
class SensorSet:
    pressure_nodes: list
    flow_links: list
    tank_nodes: list


def choose_sensors(base: Baseline, k_pressure: int, rng) -> SensorSet:
    """Spread pressure sensors over the network (farthest-point sampling on hop distance)."""
    G = base.wn.to_graph().to_undirected()
    cand = list(base.eligible)
    first = str(rng.choice(cand))
    chosen = [first]
    dist = nx.single_source_shortest_path_length(G, first)
    while len(chosen) < min(k_pressure, len(cand)):
        nxt = max(cand, key=lambda n: dist.get(n, 0))
        chosen.append(nxt)
        d2 = nx.single_source_shortest_path_length(G, nxt)
        dist = {n: min(dist.get(n, 1e9), d2.get(n, 1e9)) for n in set(dist) | set(d2)}
    return SensorSet(chosen, list(base.wn.pump_name_list), list(base.wn.tank_name_list))


def observe(res, s: SensorSet, rng, p_sigma=0.5, q_rel=0.02, level_sigma=0.05, dropout=0.01) -> pd.DataFrame:
    """Noisy, partially missing telemetry derived from the TRUE simulation results."""
    t = res.node["pressure"].index
    cols = {}
    for n in s.pressure_nodes:
        cols[f"P:{n}"] = res.node["pressure"][n].values + rng.normal(0, p_sigma, len(t))
    for l in s.flow_links:
        q = res.link["flowrate"][l].values * 1000.0                      # L/s
        cols[f"Q:{l}"] = q * (1 + rng.normal(0, q_rel, len(t)))
    for n in s.tank_nodes:
        cols[f"L:{n}"] = res.node["pressure"][n].values + rng.normal(0, level_sigma, len(t))
    df = pd.DataFrame(cols, index=t)
    mask = rng.random(df.shape) < dropout
    return df.mask(mask)


def expected_telemetry(res, s: SensorSet) -> pd.DataFrame:
    """Noise-free telemetry channels (same layout as `observe`) from a simulation: the twin's expectation."""
    t = res.node["pressure"].index
    cols = {}
    for n in s.pressure_nodes:
        cols[f"P:{n}"] = res.node["pressure"][n].values
    for l in s.flow_links:
        cols[f"Q:{l}"] = res.link["flowrate"][l].values * 1000.0
    for n in s.tank_nodes:
        cols[f"L:{n}"] = res.node["pressure"][n].values
    return pd.DataFrame(cols, index=t)
