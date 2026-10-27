"""Closed-loop staged response vs single-shot policies on leak incidents (Net3 by default). All outcomes are scored on the TRUE plant.
usage: python scripts/closed_loop.py --sensors 25 --n-leak 20"""
import argparse, os, sys, time, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("wntr").setLevel(logging.ERROR)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wdnv.config import Config
from wdnv.verifier import Model, make_day, run_incident, evaluate, select, model_with_incident
from wdnv.incidents import generate_incidents
from wdnv.actions import HOLD, enumerate_candidates
from wdnv.telemetry import observe
from wdnv.diagnosis import belief_set, _lib_pipes
from wdnv.policy import build_table, select_verified
from wdnv.pipeline import build_stack
from wdnv.regions import RegionIndex
from wdnv.closedloop import run_closed_loop, ClosedLoopParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="Net3"); ap.add_argument("--sensors", type=int, default=25)
    ap.add_argument("--n-leak", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--value-scale", type=float, default=1.0); ap.add_argument("--leak-frac", nargs=2, type=float, default=[0.03, 0.12]); ap.add_argument("--max-stages", type=int, default=3)
    ap.add_argument("--stop-conf", type=float, default=0.6); ap.add_argument("--min-leak-prob", type=float, default=0.25); ap.add_argument("--min-severity", type=float, default=0.002)
    ap.add_argument("--slice", nargs=2, type=int, default=None, help="process only incidents [lo, hi) of the fixed incident list (resumable runs)")
    ap.add_argument("--merge", default=None, help="glob of per-slice incident CSVs to summarise (no simulation)")
    ap.add_argument("--out", default="results/closedloop")
    a = ap.parse_args(); cfg = Config(); cfg.leak_frac_range = tuple(a.leak_frac)
    if a.merge:
        import glob
        R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(a.merge))], ignore_index=True)
        R.to_csv(f"{a.out}_K{a.sensors}_incidents.csv", index=False); summarise(R, a); return
    t0 = time.time(); st = build_stack(a.network, cfg, a.seed, a.sensors)
    print(f"[{a.network} K={a.sensors}] stack ready in {time.time()-t0:.0f}s: T_type={st.dg.T_type:.1f} T_loc={st.dg.T_loc:.1f}", flush=True)
    ri = RegionIndex(st.plant.wn, _lib_pipes(st.plant.wn)); daily = st.plant.base.mean_demand_m3s * 86400
    params = ClosedLoopParams(max_stages=a.max_stages, value_scale=a.value_scale, stop_conf=a.stop_conf, min_leak_prob=a.min_leak_prob)
    p0 = ClosedLoopParams(max_stages=0, stop_conf=a.stop_conf, min_leak_prob=a.min_leak_prob)
    pf = ClosedLoopParams(max_stages=a.max_stages, force=True, stop_conf=a.stop_conf, min_leak_prob=a.min_leak_prob)
    rows = []
    incs = generate_incidents(st.plant.base, cfg, np.random.default_rng(a.seed + 2000), {"LEAK": a.n_leak, "PUMP_FAILURE": 0, "DEMAND_SURGE": 0})
    for idx, inc in enumerate(incs):
        if a.slice and not (a.slice[0] <= idx < a.slice[1]):
            continue
        rng = np.random.default_rng([a.seed + 77, idx])
        wn_day = make_day(st.plant.wn, rng); day = Model.build(wn_day, cfg); seed_obs = int(rng.integers(1 << 30))
        k_alarm = int(inc.t_start // cfg.step_s) + st.latency; t_act = (k_alarm + st.win.wait + 1) * cfg.step_s
        hold = evaluate(day, inc, HOLD, t_act, cfg)
        if hold.status != "ok" or hold.objective < a.min_severity * daily:
            continue
        def tel_fn(history, wn_day=wn_day, inc=inc, seed_obs=seed_obs):
            return observe(run_incident(wn_day, inc, cfg, history=history)[1], st.sens, np.random.default_rng(seed_obs), p_sigma=st.noise_sd)
        pool_o = enumerate_candidates(model_with_incident(day, inc, cfg), inc, cfg)
        orc, _ = select([evaluate(day, inc, c, t_act, cfg) for c in pool_o], hold, cfg); orc = orc if orc is not None else hold
        d = st.dg.diagnose(tel_fn([]).iloc[: k_alarm + st.win.wait + 1], k_alarm)
        beliefs = belief_set(st.dg, d, cfg); tab = build_table(st.twin, beliefs, t_act, cfg)
        i, _ = select_verified(tab, cfg); single = evaluate(day, inc, tab.pool[i] if i is not None else HOLD, t_act, cfg)
        r = dict(incident=inc.id, true_pipe=inc.params["pipe"], leak_frac=inc.params.get("target_frac_of_demand"), hold_cost=hold.objective,
                 oracle_cost=orc.objective, single_cost=single.objective, single_violation=not single.hard_safe,
                 single_leak=single.metrics["leak_m3"], single_unmet=single.metrics["unmet_m3"],
                 hold_leak=hold.metrics["leak_m3"], oracle_leak=orc.metrics["leak_m3"], oracle_unmet=orc.metrics["unmet_m3"])
        for name, pr in (("nostage", p0), ("closed", params), ("forced", pf)):
            res = run_closed_loop(st.dg, st.twin, ri, cfg, pr, k_alarm, tel_fn, st.probe_noise.sigma)
            o = evaluate(day, inc, HOLD, res.t_final, cfg, history=res.history)
            top = max(res.final_post, key=res.final_post.get) if res.final_post else None
            r.update({f"{name}_cost": o.objective, f"{name}_leak": o.metrics["leak_m3"], f"{name}_unmet": o.metrics["unmet_m3"],
                      f"{name}_ops": o.metrics["ops"], f"{name}_violation": not o.hard_safe, f"{name}_stages": len(res.stages),
                      f"{name}_top_correct": top == inc.params["pipe"], f"{name}_top_p": res.final_post.get(top, np.nan) if top else np.nan,
                      f"{name}_t_final_h": (res.t_final - inc.t_start) / 3600, f"{name}_action": res.final_action.key()[:70], f"{name}_stagelog": str(res.stages)})
        rows.append(r)
        print(f"  {inc.id} pipe {inc.params['pipe']}: hold {hold.objective:7.0f} oracle {orc.objective:7.0f} single {single.objective:7.0f} "
              f"nostage {r['nostage_cost']:7.0f} closed {r['closed_cost']:7.0f} (stages {r['closed_stages']}) forced {r['forced_cost']:7.0f} (stages {r['forced_stages']}, top correct {r['forced_top_correct']})", flush=True)
    R = pd.DataFrame(rows); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tag = f"_s{a.slice[0]}-{a.slice[1]}" if a.slice else ""
    R.to_csv(f"{a.out}_K{a.sensors}{tag}_incidents.csv", index=False)
    summarise(R, a, t0)


def summarise(R, a, t0=None):
    base, orc = R.hold_cost.sum(), R.oracle_cost.sum(); out = []
    for name, col in (("HOLD", "hold_cost"), ("single-shot robust", "single_cost"), ("closed loop, no probing", "nostage_cost"),
                      ("closed loop (probe+refine)", "closed_cost"), ("closed loop, forced probing", "forced_cost"), ("oracle (true location, immediate)", "oracle_cost")):
        c = R[col]; v = R[{"single_cost": "single_violation", "nostage_cost": "nostage_violation", "closed_cost": "closed_violation", "forced_cost": "forced_violation"}.get(col, "single_violation")] if col not in ("hold_cost", "oracle_cost") else pd.Series([False] * len(R))
        out.append(dict(policy=name, n=len(R), cost_vs_hold_pct=round(100 * (c.sum() / base - 1), 1), share_of_oracle_gain_pct=round(100 * (base - c.sum()) / max(base - orc, 1e-9), 1),
                        worse_than_hold_pct=round(100 * float((c > R.hold_cost * 1.05 + 5).mean()), 1), violation_pct=round(100 * float(v.mean()), 1)))
    S = pd.DataFrame(out); S.to_csv(f"{a.out}_K{a.sensors}_summary.csv", index=False); print(S.to_string(index=False))
    print(f"forced probing: mean stages {R.forced_stages.mean():.2f}; final top pipe correct {100*R.forced_top_correct.mean():.0f}%; leak {R.forced_leak.sum():.0f} m3, unmet {R.forced_unmet.sum():.0f} m3")
    print(f"closed loop: mean stages {R.closed_stages.mean():.2f}; final top pipe correct {100*R.closed_top_correct.mean():.0f}% (no-probe: {100*R.nostage_top_correct.mean():.0f}%); "
          f"mean time to final action {R.closed_t_final_h.mean():.1f}h after onset; mean extra valve ops {R.closed_ops.mean():.1f}")
    print(f"leak volume (m3): hold {R.hold_leak.sum():.0f} | single {R.single_leak.sum():.0f} | no-probe {R.nostage_leak.sum():.0f} | closed {R.closed_leak.sum():.0f} | oracle {R.oracle_leak.sum():.0f}; "
          f"unmet demand (m3): single {R.single_unmet.sum():.0f} | closed {R.closed_unmet.sum():.0f} | oracle {R.oracle_unmet.sum():.0f}")
    if t0: print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
