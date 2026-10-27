from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Config:
    # --- simulation timing -------------------------------------------------
    step_s: int = 900                 # hydraulic = pattern = report step (15 min)
    horizon_h: int = 24
    # --- pressure-dependent demand (EPANET 2.2 PDA) ------------------------
    pdd_min_m: float = 7.0
    pdd_req_m: float = 20.0
    pdd_exp: float = 0.5
    # --- leak orifice ------------------------------------------------------
    leak_cd: float = 0.75
    leak_frac_range: tuple = (0.03, 0.12)    # leak flow as a share of mean system demand (sampler)
    # --- constraints (all evaluated on 'eligible' nodes, see Baseline) -----
    service_pressure_m: float = 20.0
    hard_min_pressure_m: float = 10.0     # hard limit on eligible nodes after the incident
    eligible_margin_m: float = 5.0        # eligible = baseline min pressure >= hard_min + margin
    hard_envelope_excess_m: float = 15.0  # pressure above the node's own baseline maximum (over the day); was 10 m until the sensitivity check in README
    tank_margin_m: float = 0.05           # tank counts as emptied if level <= min_level + margin
    # --- objective: total water not usefully delivered (m3) + ops tie-break
    w_unmet: float = 1.0
    w_leak: float = 1.0
    ops_penalty_m3: float = 5.0           # tie-break cost per valve/pump operation
    # --- action limits -----------------------------------------------------
    max_valve_ops: int = 6                # valve operations for a single isolation action
    max_probe_ops: int = 20               # valve operations (close + open) in one probe / re-configuration step
    speed_range: tuple = (0.7, 1.2)
    # --- bookkeeping --------------------------------------------------------
    sink_prefix: str = "LKR_"
    leak_prefix: str = "LKP_"
    break_prefix: str = "BRK_"
