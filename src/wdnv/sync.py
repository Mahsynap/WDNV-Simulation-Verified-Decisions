"""SCADA-synchronised twin.

The twin's automatic controls (time schedules and tank-level hysteresis) are *emulated on the observed tank levels* and the
resulting link statuses are forced into the twin, which is then run once per telemetry record. This is how utility
online models avoid desynchronising from the plant when level-triggered pumps switch at slightly different moments.
A failed pump is flagged naturally: the controls say it should run, the telemetry shows no flow."""
from __future__ import annotations
from collections import defaultdict
import copy
import operator
import numpy as np
import pandas as pd
from wntr.network import controls as C
from wntr.network.elements import LinkStatus

from .config import Config
from .network import add_time_control, drop_controls_for, run_epanet
from .telemetry import SensorSet, expected_telemetry
from .incidents import apply_incident
from .actions import apply_action

_OPS = {"lt": operator.lt, "gt": operator.gt, "le": operator.le, "ge": operator.ge, "eq": operator.eq}


class SyncTwin:
    def __init__(self, twin, sensors: SensorSet, cfg: Config):
        self.twin, self.sensors, self.cfg = twin, sensors, cfg
        self.rules = defaultdict(list)              # link -> [(kind, params, status)]
        self.unsupported = []
        for cname, c in twin.wn.controls():
            cond = c.condition
            for a in c.actions():
                tgt, attr = a.target()
                if attr != "status":
                    self.unsupported.append(cname); continue
                if isinstance(cond, C.SimTimeCondition) and cond._relation.name == "eq":
                    self.rules[tgt.name].append(("time", float(cond._threshold), int(a._value)))
                elif isinstance(cond, C.TankLevelCondition):
                    self.rules[tgt.name].append(("level", (cond._source_obj.name, cond._relation.name, float(cond._threshold)), int(a._value)))
                else:
                    self.unsupported.append(cname)
        tanks = list(twin.wn.tank_name_list)
        self.free_levels = twin.base.results.node["pressure"][tanks] if tanks else pd.DataFrame()

    def _levels(self, tel: pd.DataFrame) -> pd.DataFrame:
        lv = {}
        for t in self.twin.wn.tank_name_list:
            col = f"L:{t}"
            if col in tel.columns:
                lv[t] = tel[col].ffill().bfill().values
            else:
                lv[t] = self.free_levels[t].values[: len(tel)]
        return pd.DataFrame(lv, index=tel.index)

    def observed_status(self, tel: pd.DataFrame) -> dict:
        """on/off of each flow-instrumented pump from the telemetry (forward-filled over dropouts)."""
        out = {}
        for l in self.sensors.flow_links:
            col = f"Q:{l}"
            if col not in tel.columns:
                continue
            nominal = float(np.abs(self.twin.base.results.link["flowrate"][l]).max() * 1000.0)
            thr = max(2.0, 0.03 * nominal)
            q = tel[col].ffill().bfill().abs().values
            out[l] = (q > thr).astype(int)
        return out

    def emulated_statuses(self, tel: pd.DataFrame) -> dict:
        times = tel.index.values.astype(float)
        L = self._levels(tel)
        out = {}
        for link, rules in self.rules.items():
            s = 1 if self.twin.wn.get_link(link).initial_status == LinkStatus.Open else 0
            arr = np.zeros(len(times), dtype=int)
            for k, t in enumerate(times):
                prev = times[k - 1] if k > 0 else -1.0
                for kind, p, st in rules:
                    if kind == "time":
                        if prev < p <= t:
                            s = st
                    else:
                        tank, rel, thr = p
                        if _OPS[rel](float(L[tank].values[k]), thr):
                            s = st
                arr[k] = s
            out[link] = arr
        return out

    def statuses(self, tel: pd.DataFrame) -> dict:
        """statuses forced into the twin: observed for instrumented pumps, control-emulated otherwise."""
        sts = self.emulated_statuses(tel)
        sts.update({l: a for l, a in self.observed_status(tel).items() if l in sts or True})
        return sts

    def determinate_statuses(self, tel: pd.DataFrame) -> dict:
        """Status the controls *demand* at each step, or -1 when undetermined. A level rule is only decisive while the observed
        level is currently beyond its threshold (inside a hysteresis band, or when a threshold was crossed between two
        samples, the pump status cannot be inferred from sampled levels)."""
        em = self.emulated_statuses(tel)
        L = self._levels(tel)
        out = {}
        for link, rules in self.rules.items():
            arr = np.full(len(tel), -1, dtype=int)
            has_time = any(r[0] == "time" for r in rules)
            for k in range(len(tel)):
                hit = None
                for kind, p, st in rules:
                    if kind == "level" and _OPS[p[1]](float(L[p[0]].values[k]), p[2]):
                        hit = st
                if hit is not None:
                    arr[k] = hit
                elif has_time:
                    arr[k] = em[link][k]
            out[link] = arr
        # instrumented pumps without any control (e.g. an always-on pump): the demanded status is simply its initial status
        for l in self.sensors.flow_links:
            if l not in out and l in self.twin.wn.pump_name_list:
                out[l] = np.full(len(tel), 1 if self.twin.wn.get_link(l).initial_status == LinkStatus.Open else 0, dtype=int)
        return out

    def status_gap(self, tel: pd.DataFrame) -> pd.DataFrame:
        """observed - demanded status per pump where the controls leave no ambiguity
        (-1: should run but does not, +1: runs but should not, 0: consistent or undetermined)."""
        det, ob = self.determinate_statuses(tel), self.observed_status(tel)
        cols = {}
        for l in ob:
            if l in det:
                cols[l] = np.where(det[l] >= 0, ob[l] - det[l], 0)
        return pd.DataFrame(cols, index=tel.index)

    def status_alarm(self, tel: pd.DataFrame, persist: int = 3):
        """first step at which some pump has been in persistent disagreement for `persist` consecutive steps."""
        g = self.status_gap(tel)
        best = None
        for l in g.columns:
            v = (g[l].values != 0).astype(int)
            run = 0
            for k, x in enumerate(v):
                run = run + 1 if x else 0
                if run >= persist:
                    best = k if best is None else min(best, k); break
        return best

    def forced_model(self, tel: pd.DataFrame):
        """twin copy whose pump/link statuses follow the observations in `tel`."""
        sts = self.statuses(tel)
        wn = copy.deepcopy(self.twin.wn)
        times = tel.index.values
        horizon = wn.options.time.duration
        for link, arr in sts.items():
            drop_controls_for(wn, link)
            wn.get_link(link).initial_status = LinkStatus.Open if arr[0] else LinkStatus.Closed
            for k in range(1, len(arr)):
                if arr[k] != arr[k - 1] and times[k] < horizon:
                    add_time_control(wn, int(times[k]), link, LinkStatus.Open if arr[k] else LinkStatus.Closed, f"sync_{link}_{k}")
        return wn

    def expected(self, tel: pd.DataFrame) -> pd.DataFrame:
        return expected_telemetry(run_epanet(self.forced_model(tel)), self.sensors).reindex(tel.index)

    def expected_batch(self, tel: pd.DataFrame, variants: list) -> list:
        """Expected telemetry under several (incident, actions) variants on the same synchronised twin.
        `variants` = [(Incident | None, [(t, Action), ...]), ...]; the simulation stops at the last observed step."""
        base = self.forced_model(tel)
        base.options.time.duration = int(tel.index.values[-1])
        out = []
        for inc, acts in variants:
            wn = copy.deepcopy(base)
            if inc is not None:
                apply_incident(wn, inc, self.cfg)
            for t, a in acts:
                apply_action(wn, a, t, self.cfg)
            out.append(expected_telemetry(run_epanet(wn), self.sensors).reindex(tel.index))
        return out
