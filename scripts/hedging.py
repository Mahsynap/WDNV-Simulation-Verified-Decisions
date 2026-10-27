"""Hedging breadth: isolate the top-k most likely pipes at once (k = 1..kmax), unverified, and twin-verified choice of k.
One CSV per (K, seed). Summarise with scripts/hedging_summary.py.   usage: python scripts/hedging.py --sensors 25 --seeds 0 1"""
import argparse, os, sys, time, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("wntr").setLevel(logging.ERROR)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wdnv.config import Config
from wdnv.verifier import Model, make_day, run_incident, evaluate, select, model_with_incident
from wdnv.incidents import Incident, generate_incidents
from wdnv.actions import Action, HOLD, enumerate_candidates
from wdnv.telemetry import observe
from wdnv.metrics import hard_violations
from wdnv.pipeline import build_stack
from wdnv.policy import Table, select_verified

THRS = (10.0, 15.0, 20.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="Net3"); ap.add_argument("--sensors", type=int, default=25)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0]); ap.add_argument("--n-leak", type=int, default=14)
    ap.add_argument("--kmax", type=int, default=6); ap.add_argument("--min-severity", type=float, default=0.002)
    ap.add_argument("--out", default="results/hedging")
    a = ap.parse_args(); cfg = Config()
    for seed in a.seeds:
        t0 = time.time(); st = build_stack(a.network, cfg, seed, a.sensors); lib = st.lib
        daily = st.plant.base.mean_demand_m3s * 86400; rows = []
        leak_idx = {h.params["pipe"]: i for i, h in enumerate(lib.hyps) if h.kind == "LEAK"}
        incs = generate_incidents(st.plant.base, cfg, np.random.default_rng(seed + 2000), {"LEAK": a.n_leak, "PUMP_FAILURE": 0, "DEMAND_SURGE": 0})
        for idx, inc in enumerate(incs):
            rng = np.random.default_rng([seed + 77, idx]); wn_day = make_day(st.plant.wn, rng); day = Model.build(wn_day, cfg)
            seed_obs = int(rng.integers(1 << 30)); k_alarm = int(inc.t_start // cfg.step_s) + st.latency
            k_dec = k_alarm + st.win.wait; t_act = (k_dec + 1) * cfg.step_s
            hold = evaluate(day, inc, HOLD, t_act, cfg)
            if hold.status != "ok" or hold.objective < a.min_severity * daily:
                continue
            tel = observe(run_incident(wn_day, inc, cfg)[1], st.sens, np.random.default_rng(seed_obs), p_sigma=st.noise_sd)
            d = st.dg.diagnose(tel.iloc[: k_dec + 1], k_alarm)
            post = d.leak_probs(lib, st.dg.T_loc); order = sorted(post, key=post.get, reverse=True)
            true_rank = order.index(inc.params["pipe"]) + 1 if inc.params["pipe"] in order else 10 ** 6
            top = order[: a.kmax]; t_est = max(0, (k_alarm - 2) * cfg.step_s)
            pool_o = enumerate_candidates(model_with_incident(day, inc, cfg), inc, cfg)
            orc, _ = select([evaluate(day, inc, c, t_act, cfg) for c in pool_o], hold, cfg); orc = orc if orc is not None else hold
            acts = [HOLD] + [Action("CLOSE_PIPES", {"pipes": top[:k]}) for k in range(1, a.kmax + 1)]
            plant = [hold] + [evaluate(day, inc, acts[k], t_act, cfg) for k in range(1, a.kmax + 1)]
            z = sum(post[x] for x in top)
            bel = [(Incident("BEL", "LEAK", t_est, {"pipe": x, "area": max(float(d.amp[leak_idx[x]]), 0.05) * lib.hyps[leak_idx[x]].amp_unit, "cd": cfg.leak_cd}), post[x] / z) for x in top]
            outs = [[evaluate(st.twin, b, act, t_act, cfg) for b, _ in bel] for act in acts]
            tab = Table(acts, np.array([w for _, w in bel]), outs)
            r = dict(seed=seed, K=a.sensors, idx=idx, pipe=inc.params["pipe"], leak_frac=inc.params.get("target_frac_of_demand"), true_rank=true_rank,
                     hold_cost=hold.objective, oracle_cost=orc.objective, top1_p=post[top[0]])
            for k in range(0, a.kmax + 1):
                o = plant[k]; r[f"cost_k{k}"] = o.objective if o.status == "ok" else np.nan
                r[f"unmet_k{k}"] = o.metrics.get("unmet_m3", np.nan); r[f"leak_k{k}"] = o.metrics.get("leak_m3", np.nan)
                for thr in THRS:
                    c2 = Config(); c2.hard_envelope_excess_m = thr
                    r[f"viol{int(thr)}_k{k}"] = bool(o.status == "ok" and hard_violations(o.metrics, c2))
            for thr in THRS:
                c2 = Config(); c2.hard_envelope_excess_m = thr
                for row in tab.outs:
                    for o in row:
                        if o.status == "ok":
                            o.violations = hard_violations(o.metrics, c2)
                i, mode = select_verified(tab, c2)
                k = i if i is not None else 0
                r[f"ver{int(thr)}_k"] = k; r[f"ver{int(thr)}_mode"] = mode
            rows.append(r)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(f"{a.out}_K{a.sensors}_seed{seed}.csv", index=False)
        print(f"[seed {seed} K={a.sensors}] {len(rows)} incidents, top-1 exact {np.mean([r['true_rank']==1 for r in rows]):.2f}, "
              f"top-3 {np.mean([r['true_rank']<=3 for r in rows]):.2f}, top-6 {np.mean([r['true_rank']<=6 for r in rows]):.2f} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
