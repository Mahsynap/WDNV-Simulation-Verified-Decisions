"""Observation packet + prompts. The packet is a pure function of telemetry and diagnosis: it has NO access to the true incident."""
from __future__ import annotations
import json
import numpy as np

from .config import Config
from .diagnosis import channel_names, TYPES
from .incidents import G_ACC
from .llm import PACKET_OPEN, PACKET_CLOSE

SYSTEM_PROMPT = """You are a decision-support assistant for the operators of a water distribution network. An alarm has been raised.
You receive an observation packet (telemetry findings and a statistical diagnosis with calibrated probabilities) and must recommend actions.
Rules:
- Answer with ONE JSON object and nothing else, with exactly these keys:
  "assessment" (short string), "most_likely_incident" ("LEAK" | "PUMP_FAILURE" | "DEMAND_SURGE" | "UNCERTAIN"),
  "actions" (list of at most 3 objects, best first), "escalate_to_operator" (true/false).
- Each action object: {"kind": ..., "params": ..., "rationale": short string, "confidence": number 0-1}. Allowed kinds and params:
  "HOLD" {}                                           (do nothing now)
  "CLOSE_PIPES" {"pipes": [pipe ids]}                 (isolate by closing the listed pipes; at most 6 pipes; closing cuts supply to dependent customers)
  "SET_PUMP_STATUS" {"pump": id, "status": "OPEN"|"CLOSED"}
  "SET_PUMP_SPEED" {"pump": id, "speed": number}      (only for pumps with speed_adjustable true; relative speed within the allowed range)
- Use only asset ids that appear in the packet (pumps, candidate pipes and their adjacent pipes). Never invent ids.
- If the evidence is too uncertain to justify an action, return an empty "actions" list and set "escalate_to_operator" to true.
"""


def _round(x, n=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def build_packet(tel, dg, d, twin, cfg: Config, k_alarm: int, top_n: int = 8) -> dict:
    sens, lib, step = dg.sensors, dg.lib, cfg.step_s
    k_dec = k_alarm + dg.win.wait
    names = channel_names(sens)
    pcols = [i for i, n in enumerate(names) if n.startswith("P:")]
    chg = sorted(((names[i][2:], float(d.d_obs[i])) for i in pcols if not np.isnan(d.d_obs[i])), key=lambda x: -abs(x[1]))
    findings = [{"node": n, "pressure_change_m": _round(v, 2)} for n, v in chg[:8]]
    det = dg.sync.determinate_statuses(tel.iloc[: k_dec + 1])
    obs = dg.sync.observed_status(tel.iloc[: k_dec + 1])
    pumps = []
    for p in twin.wn.pump_name_list:
        ln = twin.wn.get_link(p)
        q = tel[f"Q:{p}"].iloc[max(0, k_dec - 1): k_dec + 1].mean() if f"Q:{p}" in tel.columns else float("nan")
        dem = det.get(p, [-1])[-1] if p in det else -1
        pumps.append({"id": p, "type": ln.pump_type, "speed_adjustable": ln.pump_type == "HEAD",
                      "observed_status": "on" if (p in obs and obs[p][-1] == 1) else "off", "demanded_status": {1: "on", 0: "off"}.get(int(dem), "undetermined"),
                      "observed_flow_Ls": _round(q, 1)})
    tanks = []
    for t in twin.wn.tank_name_list:
        col = f"L:{t}"
        if col in tel.columns:
            lv = tel[col].ffill().iloc[: k_dec + 1].values
            tanks.append({"id": t, "level_m": _round(lv[-1], 2), "change_last_hour_m": _round(lv[-1] - lv[max(0, len(lv) - 5)], 2),
                          "min_level_m": _round(twin.wn.get_node(t).min_level, 2), "max_level_m": _round(twin.wn.get_node(t).max_level, 2)})
    tp = d.type_probs(lib, dg.T_type)
    lp = d.leak_probs(lib, dg.T_loc)
    leak_idx = {h.params["pipe"]: i for i, h in enumerate(lib.hyps) if h.kind == "LEAK"}
    cands = []
    for pipe in sorted(lp, key=lp.get, reverse=True)[:top_n]:
        ln = twin.wn.get_link(pipe)
        adj = sorted({l for n in (ln.start_node_name, ln.end_node_name) for l in twin.wn.get_links_for_node(n)
                      if l != pipe and twin.wn.get_link(l).link_type == "Pipe" and not l.endswith("__b")})[:6]
        i = leak_idx[pipe]
        area = max(float(d.amp[i]), 0.05) * lib.hyps[i].amp_unit
        cands.append({"pipe": pipe, "probability": _round(lp[pipe]), "end_nodes": [ln.start_node_name, ln.end_node_name], "adjacent_pipes": adj,
                      "estimated_leak_Ls_if_leak": _round(area * cfg.leak_cd * np.sqrt(2 * G_ACC * 40.0) * 1000.0, 1)})
    hh, mm = divmod(int(k_alarm * step // 60), 60)
    return {"alarm_time_of_day": f"{hh:02d}:{mm:02d}", "minutes_of_data_after_alarm": int(dg.win.wait * step // 60),
            "network": {"n_pipes": len(twin.wn.pipe_name_list), "n_junctions": len(twin.wn.junction_name_list), "n_pressure_sensors": len(sens.pressure_nodes)},
            "pumps": pumps, "tanks": tanks, "sensor_findings": {"largest_pressure_changes": findings,
                                                                 "sensors_with_drop_over_1m": int(sum(1 for _, v in chg if v < -1.0))},
            "diagnosis": {"type_probabilities": {k: _round(v) for k, v in tp.items()}, "top_leak_candidates": cands,
                          "note": "calibrated probabilities of a statistical diagnosis; the true pipe is outside the candidate list in many incidents"},
            "operating_rules": {"min_pressure_m_for_connected_customers": cfg.hard_min_pressure_m, "max_pipes_per_closure_action": cfg.max_valve_ops,
                                "speed_range": list(cfg.speed_range)}}


def retrieval_query(pk: dict) -> str:
    """Deterministic text query built ONLY from what the packet shows."""
    parts = []
    if any(p["demanded_status"] == "on" and p["observed_status"] == "off" for p in pk["pumps"]):
        parts.append("pump status off while controls demand it on standby pump trip response")
    tp = pk["diagnosis"]["type_probabilities"]
    cands = pk["diagnosis"]["top_leak_candidates"]
    if tp.get("LEAK", 0) >= 0.3:
        parts.append("persistent pressure drop near one location leak isolation of a suspected pipe")
    if tp.get("DEMAND_SURGE", 0) >= 0.3:
        parts.append("pressure falls across many sensors demand surge versus leak")
    if cands and cands[0]["probability"] is not None and cands[0]["probability"] < 0.3:
        parts.append("ambiguous evidence several candidate pipes similar probability uncertain location escalate")
    if any((t["change_last_hour_m"] or 0) < -0.3 for t in pk["tanks"]):
        parts.append("tank level falling storage")
    parts.append("minimum pressure valve operations constraints")
    return " ".join(parts)


def render_prompts(pk: dict, chunks=None):
    user = "OBSERVATION PACKET (JSON):\n" + PACKET_OPEN + "\n" + json.dumps(pk, indent=1, sort_keys=True) + "\n" + PACKET_CLOSE + "\n"
    if chunks:
        user += "\nREFERENCE NOTES (general operating guidance; may or may not apply):\n" + "\n".join(f"[{c.id}] {c.title}: {c.text}" for c in chunks) + "\n"
    user += "\nRecommend actions now. Reply with the JSON object only."
    return SYSTEM_PROMPT, user
