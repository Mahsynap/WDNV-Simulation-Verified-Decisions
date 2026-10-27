"""Beliefs -> candidate actions -> policies, scored on the TRUE plant.
Policies: HOLD | naive playbook on the top belief | top-1 belief + twin verification | robust verification over the belief set |
robust + decision-theoretic abstention (defer when expected regret is large; an expert then acts after a review delay with the true diagnosis)."""
import argparse, os, sys, time, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("wntr").setLevel(logging.ERROR)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wdnv.config import Config
from wdnv.network import load_network
from wdnv.verifier import Model, make_twin, make_day, run_incident, evaluate, select, model_with_incident
from wdnv.incidents import generate_incidents
from wdnv.actions import HOLD, enumerate_candidates
from wdnv.telemetry import choose_sensors, observe
from wdnv.sync import SyncTwin
from wdnv.diagnosis import build_library, NoiseModel, Diagnoser, WindowSpec, fit_temperature, belief_set
from wdnv.evaluation import PipeGraph, true_hypothesis_index
from wdnv.policy import build_table, select_verified, naive_response

EPS = (0.10, 0.25, 0.50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=["Net3"]); ap.add_argument("--sensors", nargs="+", type=int, default=[10, 25])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--latency-steps", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.5); ap.add_argument("--twin-demand-sigma", type=float, default=0.10)
    ap.add_argument("--cal", nargs=3, type=int, default=[40, 12, 20]); ap.add_argument("--test", nargs=3, type=int, default=[40, 10, 15])
    ap.add_argument("--review-delay-steps", type=int, default=8)
    ap.add_argument("--min-severity", type=float, default=0.002)
    ap.add_argument("--out", default="results/e2e")
    a = ap.parse_args(); cfg, win = Config(), WindowSpec()
    rows = []
    for net in a.networks:
        rng = np.random.default_rng(a.seed)
        plant = Model.build(load_network(net, cfg), cfg); daily = plant.base.mean_demand_m3s * 86400
        twin = Model.build(make_twin(plant.wn, cfg, rng, a.twin_demand_sigma, 0.05, 0.10)[0], cfg)
        lib = build_library(twin, cfg, rng); pg = PipeGraph(plant.wn); leak_ids = [i for i, k in enumerate(lib.kinds()) if k == "LEAK"]
        print(f"[{net}] library {len(lib.hyps)} hypotheses", flush=True)
        for K in a.sensors:
            t1 = time.time(); sens = choose_sensors(twin.base, K, rng); sync = SyncTwin(twin, sens, cfg)
            def tel_of(wn_day, inc):
                _, res = run_incident(wn_day, inc, cfg); return observe(res, sens, rng, p_sigma=a.noise)
            noise = NoiseModel.fit(sync, sens, [tel_of(make_day(plant.wn, rng), None) for _ in range(30)], win, rng)
            dg = Diagnoser(lib, sens, sync, noise, win)
            cal = []
            for inc in generate_incidents(plant.base, cfg, np.random.default_rng(a.seed + 1000), dict(zip(("LEAK", "PUMP_FAILURE", "DEMAND_SURGE"), a.cal))):
                d = dg.diagnose(tel_of(make_day(plant.wn, rng), inc), int(inc.t_start // cfg.step_s) + a.latency_steps)
                ti = true_hypothesis_index(lib, inc, pg)
                if ti >= 0:
                    cal.append((inc, d, ti))
            dg.T_type = fit_temperature([d.chi2 for _, d, _ in cal], [t for _, _, t in cal])
            lc = [(d.chi2[:-1][leak_ids], leak_ids.index(t)) for inc, d, t in cal if inc.kind == "LEAK"]
            dg.T_loc = fit_temperature([c for c, _ in lc], [t for _, t in lc])
            n_inc = 0
            for inc in generate_incidents(plant.base, cfg, np.random.default_rng(a.seed + 2000), dict(zip(("LEAK", "PUMP_FAILURE", "DEMAND_SURGE"), a.test))):
                wn_day = make_day(plant.wn, rng); day = Model.build(wn_day, cfg)
                k_alarm = int(inc.t_start // cfg.step_s) + a.latency_steps
                t_act = (k_alarm + win.wait + 1) * cfg.step_s
                hold = evaluate(day, inc, HOLD, t_act, cfg)
                if hold.status != "ok" or hold.objective < a.min_severity * daily:
                    continue
                n_inc += 1
                d = dg.diagnose(tel_of(wn_day, inc), k_alarm)
                beliefs = belief_set(dg, d, cfg)
                top = max(beliefs, key=lambda b: b[1])[0]
                # oracle (true incident) on the plant
                pool_o = enumerate_candidates(model_with_incident(day, inc, cfg), inc, cfg)
                outs_o = [evaluate(day, inc, c, t_act, cfg) for c in pool_o]
                orc, _ = select(outs_o, hold, cfg); orc_out = orc if orc is not None else hold
                # expert after review delay (true diagnosis, later action)
                t_late = t_act + a.review_delay_steps * cfg.step_s
                outs_l = [evaluate(day, inc, c, t_late, cfg) for c in pool_o]
                hold_l = evaluate(day, inc, HOLD, t_late, cfg); late, _ = select(outs_l, hold_l, cfg); late_out = late if late is not None else hold_l
                # policies
                twin_tab = build_table(twin, beliefs, t_act, cfg)
                tab1 = build_table(twin, [(top, 1.0)], t_act, cfg)
                i1, m1 = select_verified(tab1, cfg); i2, m2 = select_verified(twin_tab, cfg)
                pol = {"HOLD": (HOLD, hold),
                       "naive_top1": (naive_response(top, twin, cfg), None),
                       "verified_top1": (tab1.pool[i1] if i1 is not None else HOLD, None),
                       "verified_robust": (twin_tab.pool[i2] if i2 is not None else HOLD, None)}
                r = dict(network=net, K=K, incident=inc.id, kind=inc.kind, hold_cost=hold.objective, hold_cost_share=hold.objective / daily,
                         oracle_cost=orc_out.objective, late_expert_cost=late_out.objective, n_beliefs=len(beliefs), n_pool=len(twin_tab.pool),
                         top_belief_correct=(top.kind == inc.kind and (inc.kind != "LEAK" or top.params["pipe"] == inc.params["pipe"])),
                         robust_mode=m2)
                for name, (act, o) in pol.items():
                    o = o if o is not None else evaluate(day, inc, act, t_act, cfg)
                    r[f"{name}_cost"] = o.objective if o.status == "ok" else np.nan
                    r[f"{name}_violation"] = (o.status == "ok" and not o.hard_safe)
                    r[f"{name}_action"] = act.kind if act.kind in ("HOLD",) else act.key()[:60]
                # abstention: defer when the expected regret of the robust choice is large relative to HOLD's expected cost
                if i2 is not None:
                    er = twin_tab.regret(i2); hold_i = [i for i, x in enumerate(twin_tab.pool) if x.kind == "HOLD"][0]; eh = max(twin_tab.expected(hold_i), 1.0)
                else:
                    er, eh = np.inf, 1.0
                r["exp_regret_ratio"] = er / eh
                for e in EPS:
                    defer = (i2 is None) or (er > e * eh)
                    r[f"abstain_{e}_deferred"] = defer
                    r[f"abstain_{e}_cost"] = late_out.objective if defer else r["verified_robust_cost"]
                    r[f"abstain_{e}_violation"] = False if defer else r["verified_robust_violation"]
                rows.append(r)
            print(f"[{net} K={K}] {n_inc} qualifying incidents, T_type={dg.T_type:.1f} T_loc={dg.T_loc:.1f}, {time.time()-t1:.0f}s", flush=True)
    R = pd.DataFrame(rows); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); R.to_csv(a.out + "_incidents.csv", index=False)
    pols = ["HOLD", "naive_top1", "verified_top1", "verified_robust"] + [f"abstain_{e}" for e in EPS]
    out = []
    for (net, K), g in R.groupby(["network", "K"]):
        base = g.hold_cost.sum(); orc = g.oracle_cost.sum()
        for p in pols:
            c = g[f"{p}_cost"].fillna(g.hold_cost)
            out.append(dict(network=net, K=K, policy=p, n=len(g), cost_vs_hold_pct=100 * (c.sum() / base - 1),
                            share_of_oracle_gain_recovered=100 * (base - c.sum()) / max(base - orc, 1e-9),
                            harm_rate=float((c > g.hold_cost * 1.05 + 5).mean()), violation_rate=float(g[f"{p}_violation"].mean()),
                            deferral_rate=float(g[f"{p}_deferred"].mean()) if f"{p}_deferred" in g else 0.0))
        out.append(dict(network=net, K=K, policy="oracle", n=len(g), cost_vs_hold_pct=100 * (orc / base - 1), share_of_oracle_gain_recovered=100.0, harm_rate=0.0, violation_rate=0.0, deferral_rate=0.0))
    S = pd.DataFrame(out).round(2); S.to_csv(a.out + "_summary.csv", index=False)
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
