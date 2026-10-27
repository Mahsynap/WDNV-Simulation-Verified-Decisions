"""Alarm-conditioned diagnosis experiment: calibration on one incident set, evaluation on another.
usage: python scripts/diagnosis_experiment.py --networks Net3 --sensors 10 25"""
import argparse, os, sys, time, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("wntr").setLevel(logging.ERROR)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wdnv.config import Config
from wdnv.network import load_network
from wdnv.verifier import Model, make_twin, make_day, run_incident
from wdnv.incidents import generate_incidents
from wdnv.telemetry import choose_sensors, observe
from wdnv.sync import SyncTwin
from wdnv.diagnosis import build_library, NoiseModel, Diagnoser, WindowSpec, fit_temperature, TYPES
from wdnv.evaluation import ece, credible_set, PipeGraph, true_hypothesis_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=["Net3"])
    ap.add_argument("--sensors", nargs="+", type=int, default=[10, 25])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latency-steps", type=int, default=3, help="alarm latency after onset, in 15-min steps (<5)")
    ap.add_argument("--noise", type=float, default=0.5, help="pressure sensor noise sd (m)")
    ap.add_argument("--twin-demand-sigma", type=float, default=0.10)
    ap.add_argument("--cal", nargs=3, type=int, default=[40, 12, 20], help="calibration incidents: leak pump surge")
    ap.add_argument("--test", nargs=3, type=int, default=[80, 20, 30], help="test incidents: leak pump surge")
    ap.add_argument("--out", default="results/diagnosis")
    a = ap.parse_args()
    cfg, win = Config(), WindowSpec()
    summary, per_inc = [], []
    for net in a.networks:
        rng = np.random.default_rng(a.seed)
        plant = Model.build(load_network(net, cfg), cfg)
        twin_wn, _ = make_twin(plant.wn, cfg, rng, a.twin_demand_sigma, 0.05, 0.10)
        twin = Model.build(twin_wn, cfg)
        t0 = time.time(); lib = build_library(twin, cfg, rng)
        print(f"[{net}] library: {len(lib.hyps)} hypotheses in {time.time()-t0:.0f}s", flush=True)
        pg = PipeGraph(plant.wn)
        kinds = lib.kinds(); leak_ids = [i for i, k in enumerate(kinds) if k == "LEAK"]
        for K in a.sensors:
            t1 = time.time()
            sens = choose_sensors(twin.base, K, rng); sync = SyncTwin(twin, sens, cfg)

            def telemetry(inc):
                wn = make_day(plant.wn, rng); _, res = run_incident(wn, inc, cfg)
                return observe(res, sens, rng, p_sigma=a.noise)

            noise = NoiseModel.fit(sync, sens, [telemetry(None) for _ in range(30)], win, rng)
            dg = Diagnoser(lib, sens, sync, noise, win)

            def run_set(seed, counts):
                incs = generate_incidents(plant.base, cfg, np.random.default_rng(seed), dict(zip(("LEAK", "PUMP_FAILURE", "DEMAND_SURGE"), counts)))
                out = []
                for inc in incs:
                    d = dg.diagnose(telemetry(inc), int(inc.t_start // cfg.step_s) + a.latency_steps)
                    ti = true_hypothesis_index(lib, inc, pg)
                    out.append((inc, d, ti))
                return out

            cal = [c for c in run_set(a.seed + 1000, a.cal) if c[2] >= 0]
            dg.T_type = fit_temperature([d.chi2 for _, d, _ in cal], [t for _, _, t in cal])
            leak_cal = [(d.chi2[:-1][leak_ids], leak_ids.index(t)) for inc, d, t in cal if inc.kind == "LEAK"]
            dg.T_loc = fit_temperature([c for c, _ in leak_cal], [t for _, t in leak_cal])
            test = run_set(a.seed + 2000, a.test)

            rows = []
            for inc, d, tidx in test:
                tp = d.type_probs(lib, dg.T_type)
                pred = max(TYPES, key=lambda x: tp[x])
                r = dict(network=net, K=K, incident=inc.id, kind=inc.kind, pred=pred, conf=tp[pred], p_true_type=tp[inc.kind], p_none=tp["NONE"])
                if inc.kind == "LEAK":
                    lp = d.leak_probs(lib, dg.T_loc); order = sorted(lp, key=lp.get, reverse=True); tp_ = inc.params["pipe"]
                    cs = credible_set(lp, 0.8)
                    r.update(leak_rank=order.index(tp_) + 1 if tp_ in order else 10 ** 6, leak_hop=pg.hop(order[0], tp_), leak_conf=lp[order[0]],
                             cred_size=len(cs), cred_cover=tp_ in cs, cred_cover_2hop=min(pg.hop(c, tp_) for c in cs) <= 2,
                             size_ratio=float(d.amp[tidx] * lib.hyps[tidx].amp_unit / inc.params["area"]))
                rows.append(r)
            R = pd.DataFrame(rows); per_inc.append(R)
            L = R[R.kind == "LEAK"]
            row = dict(network=net, K=K, T_type=round(dg.T_type, 1), T_loc=round(dg.T_loc, 1),
                       type_acc_leak=(L.pred == "LEAK").mean(),
                       type_acc_pump=(R[R.kind == "PUMP_FAILURE"].pred == "PUMP_FAILURE").mean(),
                       type_acc_surge=(R[R.kind == "DEMAND_SURGE"].pred == "DEMAND_SURGE").mean(),
                       type_acc_macro=np.mean([(R[R.kind == k].pred == k).mean() for k in TYPES]),
                       type_ece=ece(R.conf, R.pred == R.kind), mean_p_true_type=R.p_true_type.mean(),
                       leak_top1=(L.leak_rank == 1).mean(), leak_top5=(L.leak_rank <= 5).mean(), leak_top10=(L.leak_rank <= 10).mean(),
                       leak_median_hop=L.leak_hop.median(), leak_within1=(L.leak_hop <= 1).mean(), leak_within2=(L.leak_hop <= 2).mean(),
                       leak_ece=ece(L.leak_conf, L.leak_hop == 0), cred80_cover=L.cred_cover.mean(), cred80_size=L.cred_size.median(),
                       cred80_cover_within2=L.cred_cover_2hop.mean(), size_ratio_median=L.size_ratio.median(), n_leak_hyp=len(leak_ids), seconds=round(time.time() - t1))
            summary.append(row)
            print(f"[{net} K={K}] " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items() if k not in ("network", "K")), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame(summary).to_csv(a.out + "_summary.csv", index=False)
    pd.concat(per_inc).to_csv(a.out + "_incidents.csv", index=False)
    print("wrote", a.out + "_summary.csv")


if __name__ == "__main__":
    main()
