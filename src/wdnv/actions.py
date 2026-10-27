from __future__ import annotations
import json
from dataclasses import dataclass, field
import numpy as np
import wntr
from wntr.network import controls as C
from wntr.network.elements import LinkStatus

from .config import Config
from .network import add_time_control, limit_controls_before
from .incidents import Incident, leak_names

KINDS = ("HOLD", "CLOSE_PIPES", "SET_PIPES", "SET_PUMP_STATUS", "SET_PUMP_SPEED", "OPERATOR_REVIEW")


@dataclass
class Action:
    kind: str
    params: dict = field(default_factory=dict)

    def key(self) -> str:
        return json.dumps({"kind": self.kind, "params": self.params}, sort_keys=True)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": self.params}

    @staticmethod
    def from_dict(d) -> "Action":
        if not isinstance(d, dict):
            return Action("INVALID", {"raw": str(d)[:200]})
        return Action(str(d.get("kind", "")).strip().upper(), dict(d.get("params", {}) or {}))

    def n_ops(self) -> int:
        if self.kind == "CLOSE_PIPES":
            return len(self.params.get("pipes", []))
        if self.kind == "SET_PIPES":
            return len(self.params.get("close", [])) + len(self.params.get("open", []))
        return 1 if self.kind in ("SET_PUMP_STATUS", "SET_PUMP_SPEED") else 0

    def executable(self) -> bool:
        return self.kind in ("HOLD", "CLOSE_PIPES", "SET_PIPES", "SET_PUMP_STATUS", "SET_PUMP_SPEED")


HOLD = Action("HOLD")
REVIEW = Action("OPERATOR_REVIEW")


def validate_action(wn, a: Action, cfg: Config) -> list:
    """Return a list of error strings (empty = valid). Guards against hallucinated commands."""
    if a.kind not in KINDS:
        return [f"unsupported action kind '{a.kind}'"]
    errs = []
    if a.kind == "CLOSE_PIPES":
        pipes = a.params.get("pipes")
        if not isinstance(pipes, (list, tuple)) or not pipes:
            return ["CLOSE_PIPES needs a non-empty 'pipes' list"]
        if len(set(pipes)) != len(pipes):
            errs.append("duplicate pipes")
        if len(pipes) > cfg.max_valve_ops:
            errs.append(f"too many valve operations ({len(pipes)} > {cfg.max_valve_ops})")
        for p in pipes:
            if not isinstance(p, str) or p not in wn.link_name_list:
                errs.append(f"unknown pipe '{p}'")
            elif wn.get_link(p).link_type != "Pipe":
                errs.append(f"'{p}' is not a pipe")
            elif p.startswith(cfg.leak_prefix) or p.endswith("__b"):
                errs.append(f"'{p}' is not an operable asset")
    elif a.kind == "SET_PIPES":
        close, opn = a.params.get("close", []), a.params.get("open", [])
        if not isinstance(close, (list, tuple)) or not isinstance(opn, (list, tuple)) or not (close or opn):
            return ["SET_PIPES needs 'close' and/or 'open' lists"]
        if set(close) & set(opn) or len(set(close)) != len(close) or len(set(opn)) != len(opn):
            errs.append("duplicate or conflicting pipes")
        if len(close) + len(opn) > cfg.max_probe_ops:
            errs.append(f"too many valve operations ({len(close) + len(opn)} > {cfg.max_probe_ops})")
        for p in list(close) + list(opn):
            if not isinstance(p, str) or p not in wn.link_name_list:
                errs.append(f"unknown pipe '{p}'")
            elif wn.get_link(p).link_type != "Pipe":
                errs.append(f"'{p}' is not a pipe")
            elif p.startswith(cfg.leak_prefix) or p.endswith("__b"):
                errs.append(f"'{p}' is not an operable asset")
    elif a.kind in ("SET_PUMP_STATUS", "SET_PUMP_SPEED"):
        pump = a.params.get("pump")
        if not isinstance(pump, str) or pump not in wn.pump_name_list:
            return [f"unknown pump '{pump}'"]
        if a.kind == "SET_PUMP_STATUS":
            if str(a.params.get("status", "")).upper() not in ("OPEN", "CLOSED"):
                errs.append("status must be OPEN or CLOSED")
        else:
            v = a.params.get("speed")
            lo, hi = cfg.speed_range
            if wn.get_link(pump).pump_type != "HEAD":
                errs.append(f"pump '{pump}' is not a head pump (speed not adjustable)")
            if not isinstance(v, (int, float)) or not (lo <= float(v) <= hi):
                errs.append(f"speed must be within [{lo}, {hi}]")
    return errs


def apply_action(wn, a: Action, t_action: int, cfg: Config):
    """Schedule the action at t_action (operator overrides any automatic control on the same asset)."""
    t_action = int(t_action)
    if a.kind == "CLOSE_PIPES":
        for p0 in a.params["pipes"]:
            for p in [p0] + ([p0 + "__b"] if (p0 + "__b") in wn.link_name_list else []):   # both halves of a burst pipe
                limit_controls_before(wn, p, t_action)
                add_time_control(wn, t_action, p, LinkStatus.Closed, f"act_close_{p}")
    elif a.kind == "SET_PIPES":
        for status, names in ((LinkStatus.Closed, a.params.get("close", [])), (LinkStatus.Open, a.params.get("open", []))):
            tag = "close" if status == LinkStatus.Closed else "open"
            for p0 in names:
                for p in [p0] + ([p0 + "__b"] if (p0 + "__b") in wn.link_name_list else []):
                    limit_controls_before(wn, p, t_action)
                    add_time_control(wn, t_action, p, status, f"act_{tag}_{p}_{t_action}")
    elif a.kind == "SET_PUMP_STATUS":
        pump = a.params["pump"]
        limit_controls_before(wn, pump, t_action)             # operator override from t_action on
        st = LinkStatus.Open if str(a.params["status"]).upper() == "OPEN" else LinkStatus.Closed
        add_time_control(wn, t_action, pump, st, f"act_pump_{pump}")
    elif a.kind == "SET_PUMP_SPEED":
        pump = a.params["pump"]
        act = C.ControlAction(wn.get_link(pump), "base_speed", float(a.params["speed"]))
        name = f"act_speed_{pump}"
        wn.add_control(name, C.Control._time_control(wn, t_action, "SIM_TIME", False, act, name))
    return wn


# --------------------------------------------------------------------------
# candidate enumeration (the 'search space' an oracle or an LLM proposal is drawn from)
# --------------------------------------------------------------------------
def enumerate_candidates(wn_model, belief: Incident, cfg: Config, rng=None, n_decoys: int = 0) -> list:
    """Executable candidate actions for a *believed* incident, expressed with original asset names only.
    `wn_model` is the model with the believed incident applied (so a believed burst pipe appears split)."""
    out = [HOLD]
    for p in wn_model.pump_name_list:
        out.append(Action("SET_PUMP_STATUS", {"pump": p, "status": "OPEN"}))
        out.append(Action("SET_PUMP_STATUS", {"pump": p, "status": "CLOSED"}))
        if wn_model.get_link(p).pump_type == "HEAD":
            for s_ in (0.9, 1.1):
                out.append(Action("SET_PUMP_SPEED", {"pump": p, "speed": s_}))
    if belief is not None and belief.kind == "LEAK":
        nm = leak_names(belief, cfg)
        a, b = nm["pipe_a"], nm["pipe_b"]
        out.append(Action("CLOSE_PIPES", {"pipes": [a]}))                          # isolate the believed pipe
        ends = {wn_model.get_link(a).start_node_name, wn_model.get_link(b).end_node_name}
        ring = {l for n in ends for l in wn_model.get_links_for_node(n)
                if wn_model.get_link(l).link_type == "Pipe" and not l.startswith(cfg.leak_prefix) and not l.endswith("__b")}
        ring.add(a)
        if 1 < len(ring) <= cfg.max_valve_ops:
            out.append(Action("CLOSE_PIPES", {"pipes": sorted(ring)}))             # wider isolation around it
    seen, uniq = set(), []
    for act in out:
        if act.key() not in seen:
            seen.add(act.key()); uniq.append(act)
    return uniq
