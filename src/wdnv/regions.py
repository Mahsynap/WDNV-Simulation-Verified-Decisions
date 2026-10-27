"""Probe regions: node sets that can be cut off from every source by closing a small number of boundary pipes."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import networkx as nx


@dataclass(frozen=True)
class Region:
    nodes: frozenset
    boundary: tuple          # pipes to close
    covers: frozenset        # candidate leak pipes that become isolated (an end inside the region, or on the boundary)
    demand_m3s: float        # demand cut off while the boundary is closed


class RegionIndex:
    def __init__(self, wn, leak_pipes):
        self.wn = wn
        self.G = wn.to_graph().to_undirected()
        special = set(wn.tank_name_list) | set(wn.reservoir_name_list)
        for _, l in list(wn.pumps()) + list(wn.valves()):
            special |= {l.start_node_name, l.end_node_name}
        self.special = special
        self.ends = {n: (p.start_node_name, p.end_node_name) for n, p in wn.pipes()}
        self.node_pipes = defaultdict(list)
        for n, (a, b) in self.ends.items():
            self.node_pipes[a].append(n); self.node_pipes[b].append(n)
        self.leak_pipes = set(leak_pipes)
        self.demand = {j: sum(ts.base_value for ts in wn.get_node(j).demand_timeseries_list) for j in wn.junction_name_list}
        self._cache = {}

    def _bfs(self, s, r):
        key = (s, r)
        if key not in self._cache:
            self._cache[key] = frozenset(nx.single_source_shortest_path_length(self.G, s, cutoff=r))
        return self._cache[key]

    def regions_around(self, seeds, max_boundary: int, radii=(0, 1, 2, 3, 4)) -> list:
        out, seen = [], set()
        for s in seeds:
            for r in radii:
                R = self._bfs(s, r)
                if R in seen or (R & self.special):
                    continue
                seen.add(R)
                bnd = sorted({p for n in R for p in self.node_pipes[n] if (self.ends[p][0] in R) != (self.ends[p][1] in R)})
                if not (1 <= len(bnd) <= max_boundary):
                    continue
                covers = frozenset(p for p in self.leak_pipes if self.ends[p][0] in R or self.ends[p][1] in R)
                out.append(Region(R, tuple(bnd), covers, float(sum(self.demand.get(n, 0.0) for n in R))))
        return out
