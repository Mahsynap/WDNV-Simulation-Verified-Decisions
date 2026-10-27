"""Summarise hedging runs with a two-level (seed, incident) bootstrap.  usage: python scripts/hedging_summary.py --sensors 25"""
import argparse, glob, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--sensors", type=int, default=25); ap.add_argument("--dir", default="results")
ap.add_argument("--boot", type=int, default=3000); a = ap.parse_args()
D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{a.dir}/hedging_K{a.sensors}_seed*.csv"))], ignore_index=True)
kmax = max(int(c.split("_k")[1]) for c in D.columns if c.startswith("cost_k"))
seeds = sorted(D.seed.unique()); groups = {s: D[D.seed == s].reset_index(drop=True) for s in seeds}
print(f"K={a.sensors}: {len(seeds)} seeds, {len(D)} incidents")

def policy_cost(df, name):
    if name == "HOLD": return df.hold_cost.values
    if name == "oracle": return df.oracle_cost.values
    if name.startswith("top-"): return df[f"cost_k{int(name[4:])}"].values
    thr = int(name.split("@")[1]); k = df[f"ver{thr}_k"].values
    return np.array([df[f"cost_k{kk}"].iloc[i] for i, kk in enumerate(k)])

def policy_viol(df, name):
    if name in ("HOLD", "oracle"): return np.zeros(len(df), bool)
    if name.startswith("top-"): return df[f"viol15_k{int(name[4:])}"].values.astype(bool)
    thr = int(name.split("@")[1]); k = df[f"ver{thr}_k"].values
    return np.array([bool(df[f"viol{thr}_k{kk}"].iloc[i]) for i, kk in enumerate(k)])

def policy_unmet(df, name):
    if name == "HOLD": return df.unmet_k0.values
    if name == "oracle": return np.full(len(df), np.nan)
    if name.startswith("top-"): return df[f"unmet_k{int(name[4:])}"].values
    k = df[f"ver{int(name.split('@')[1])}_k"].values
    return np.array([df[f"unmet_k{kk}"].iloc[i] for i, kk in enumerate(k)])

names = ["HOLD"] + [f"top-{k}" for k in range(1, kmax + 1)] + ["verified@10", "verified@15", "verified@20", "oracle"]
cache = {n: {s: (policy_cost(g, n), policy_viol(g, n), policy_unmet(g, n)) for s, g in groups.items()} for n in names}
hold = {s: g.hold_cost.values for s, g in groups.items()}; orc = {s: g.oracle_cost.values for s, g in groups.items()}
rng = np.random.default_rng(0)

def stat(name, sample):            # sample: list of (seed, incident index array)
    h = np.concatenate([hold[s][ix] for s, ix in sample]); o = np.concatenate([orc[s][ix] for s, ix in sample])
    c = np.concatenate([cache[name][s][0][ix] for s, ix in sample]); v = np.concatenate([cache[name][s][1][ix] for s, ix in sample])
    share = 100 * (h.sum() - np.nansum(c)) / max(h.sum() - o.sum(), 1e-9)
    return share, 100 * (np.nansum(c) / h.sum() - 1), 100 * np.mean(c > 1.05 * h + 5), 100 * v.mean()

full = [(s, np.arange(len(groups[s]))) for s in seeds]
boot = {n: [] for n in names}
diff = []
for _ in range(a.boot):
    ss = rng.choice(seeds, size=len(seeds), replace=True)
    sample = [(s, rng.integers(0, len(groups[s]), size=len(groups[s]))) for s in ss]
    for n in names:
        boot[n].append(stat(n, sample)[0])
    diff.append(boot['verified@15'][-1] - boot['top-1'][-1])
rows = []
for n in names:
    share, cvh, worse, viol = stat(n, full); lo, hi = np.percentile(boot[n], [2.5, 97.5])
    per_seed = [stat(n, [(s, np.arange(len(groups[s])))])[0] for s in seeds]
    um = np.nanmean(np.concatenate([cache[n][s][2] for s in seeds])) if n != "oracle" else np.nan
    rows.append(dict(policy=n, cost_vs_hold_pct=round(cvh, 1), share_of_oracle_gain_pct=round(share, 1), ci95=f"[{lo:.0f}, {hi:.0f}]",
                     seed_min=round(min(per_seed)), seed_median=round(float(np.median(per_seed))), seed_max=round(max(per_seed)),
                     worse_than_hold_pct=round(worse, 1), violation_pct=round(viol, 1), mean_unmet_m3=round(um, 0) if um == um else np.nan))
S = pd.DataFrame(rows); pd.set_option("display.width", 250); print(S.to_string(index=False))
S.to_csv(f"{a.dir}/hedging_K{a.sensors}_summary.csv", index=False)
lo, hi = np.percentile(diff, [2.5, 97.5]); print(f"\npaired difference in share of oracle gain, verified@15 minus unverified top-1: {np.mean(diff):.1f} points, 95% CI [{lo:.1f}, {hi:.1f}], P(diff>0) = {np.mean(np.array(diff) > 0):.3f}")
print("\ncoverage: share of incidents where the true pipe is within the diagnosis' top-k")
cov = {k: round(100 * (D.true_rank <= k).mean(), 1) for k in (1, 2, 3, 4, 5, 6, 10, 20)}; print(cov)
print("per-seed top-1 / top-6 coverage:", {int(s): (round(100 * (groups[s].true_rank == 1).mean()), round(100 * (groups[s].true_rank <= 6).mean())) for s in seeds})
print("verified@15 chosen k distribution (0 = do nothing):", D["ver15_k"].value_counts().sort_index().to_dict())
