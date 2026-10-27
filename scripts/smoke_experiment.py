"""Smoke experiment: random incidents -> candidate pool -> oracle (plant) vs twin-selected action.
NOT a paper result: diagnosis is assumed perfect and there is no LLM yet. It exercises the full simulation/verification stack."""
import argparse, os, sys, time, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("wntr").setLevel(logging.ERROR)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wdnv.config import Config
from wdnv.network import load_network
from wdnv.incidents import generate_incidents
from wdnv.actions import HOLD, enumerate_candidates
from wdnv.verifier import Model, evaluate, select, make_twin, model_with_incident


def short(a):
    p = a.params
    if a.kind == "CLOSE_PIPES": return "CLOSE[" + ",".join(p["pipes"]) + "]"
    if a.kind == "SET_PUMP_STATUS": return f"PUMP {p['pump']} {p['status']}"
    if a.kind == "SET_PUMP_SPEED": return f"SPEED {p['pump']} x{p['speed']}"
    return a.kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=["Net3", "ky6"])
    ap.add_argument("--n-leak", type=int, default=6); ap.add_argument("--n-pump", type=int, default=3); ap.add_argument("--n-surge", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay-min", type=int, default=60, help="detection + decision delay before the action takes effect")
    ap.add_argument("--min-severity", type=float, default=0.002, help="skip incidents whose HOLD cost is below this share of daily demand")
    ap.add_argument("--demand-sigma", type=float, default=0.10); ap.add_argument("--rough-sigma", type=float, default=0.10)
    ap.add_argument("--out", default="results/smoke.csv")
    a = ap.parse_args()
    cfg = Config(); rows = []
    for net in a.networks:
        t_net = time.time()
        plant = Model.build(load_network(net, cfg), cfg)
        daily_m3 = plant.base.mean_demand_m3s * 86400
        rng = np.random.default_rng(a.seed)
        incs = generate_incidents(plant.base, cfg, rng, {"LEAK": a.n_leak, "PUMP_FAILURE": a.n_pump, "DEMAND_SURGE": a.n_surge})
        n_sims = 0
        for inc in incs:
            ta = inc.t_start + a.delay_min * 60
            hold = evaluate(plant, inc, HOLD, ta, cfg); n_sims += 1
            sev = hold.objective / daily_m3
            if hold.status != "ok" or sev < a.min_severity:
                rows.append(dict(network=net, incident=inc.id, kind=inc.kind, skipped="negligible_or_failed", severity=sev)); continue
            pool = enumerate_candidates(model_with_incident(plant, inc, cfg), inc, cfg, rng)
            plant_out = {c.key(): evaluate(plant, inc, c, ta, cfg) for c in pool}; n_sims += len(pool)
            oracle, o_mode = select(list(plant_out.values()), hold, cfg)
            twin_wn, _ = make_twin(plant.wn, cfg, rng, a.demand_sigma, rough_sigma=a.rough_sigma)
            twin = Model.build(twin_wn, cfg)
            twin_out = {c.key(): evaluate(twin, inc, c, ta, cfg) for c in pool}; n_sims += len(pool) + 1
            chosen, t_mode = select(list(twin_out.values()), twin_out[HOLD.key()], cfg)
            realized = plant_out[chosen.action.key()] if chosen is not None else hold
            # twin-vs-plant verdict agreement over all candidates (hard-safety)
            both = [(twin_out[k].hard_safe, plant_out[k].hard_safe) for k in plant_out if plant_out[k].status == "ok" and twin_out[k].status == "ok"]
            rows.append(dict(
                network=net, incident=inc.id, kind=inc.kind, t_start_h=inc.t_start / 3600, severity=round(sev, 4), skipped="",
                n_candidates=len(pool), n_invalid=sum(o.status == "invalid" for o in plant_out.values()),
                n_sim_failed=sum(o.status == "sim_failed" for o in plant_out.values()),
                n_plant_unsafe=sum(o.status == "ok" and not o.hard_safe for o in plant_out.values()),
                hold_objective_m3=round(hold.objective, 1), hold_hard_safe=hold.hard_safe,
                oracle_action=short(oracle.action) if oracle else "REVIEW", oracle_mode=o_mode,
                oracle_objective_m3=round(oracle.objective, 1) if oracle else np.nan,
                twin_action=short(chosen.action) if chosen else "REVIEW", twin_mode=t_mode,
                realized_objective_m3=round(realized.objective, 1), realized_hard_safe=realized.hard_safe,
                oracle_gain_pct=round(100 * (1 - (oracle.objective if oracle else hold.objective) / hold.objective), 1),
                realized_gain_pct=round(100 * (1 - realized.objective / hold.objective), 1),
                regret_m3=round(realized.objective - (oracle.objective if oracle else hold.objective), 1),
                twin_false_assurance=sum(t and not p for t, p in both), twin_false_alarm=sum((not t) and p for t, p in both), n_compared=len(both)))
        print(f"[{net}] {len(incs)} incidents, {n_sims} simulations, {time.time()-t_net:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); df.to_csv(a.out, index=False)
    d = df[df.skipped == ""].copy()
    if len(d):
        print("\n=== summary (smoke test, perfect diagnosis, no LLM) ===")
        g = d.groupby(["network", "kind"]).agg(n=("incident", "count"), hold_cost_m3=("hold_objective_m3", "median"),
            oracle_gain_pct=("oracle_gain_pct", "median"), twin_realized_gain_pct=("realized_gain_pct", "median"),
            regret_m3=("regret_m3", "median"), twin_review=("twin_mode", lambda s: (s == "review").sum()),
            plant_unsafe_cands=("n_plant_unsafe", "mean"), sim_failed=("n_sim_failed", "sum"), invalid=("n_invalid", "sum"),
            false_assurance=("twin_false_assurance", "sum"), false_alarm=("twin_false_alarm", "sum")).round(1)
        print(g.to_string())
    print(f"\nskipped (negligible/failed): {(df.skipped != '').sum()} of {len(df)}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
