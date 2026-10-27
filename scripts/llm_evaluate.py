"""Phase 3: evaluate model responses on the TRUE plant (rebuilds each incident deterministically and checks the packet hash).
usage: python scripts/llm_evaluate.py --packets P --truth T --responses R --out results/llm/eval_mytag"""
import argparse, json, os, sys, time, warnings, logging
from collections import defaultdict
warnings.filterwarnings("ignore"); logging.getLogger("wntr").setLevel(logging.ERROR)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wdnv.config import Config
from wdnv.incidents import generate_incidents
from wdnv.pipeline import build_stack
from wdnv.llm_pipeline import prepare_incident, packet_hash, reference_outcomes, evaluate_response
from wdnv.proposal import parse_response, Parsed
from wdnv.kb import load_kb

REL = {"LEAK": {"leak"}, "PUMP_FAILURE": {"pump"}, "DEMAND_SURGE": {"surge"}}


def boot_share(costs, hold, orc, seeds, B, rng):
    """share of oracle gain with a two-level bootstrap (seeds, then incidents within seeds)"""
    ids = {s: np.where(seeds == s)[0] for s in np.unique(seeds)}; keys = list(ids)
    def share(ix): return 100 * (hold[ix].sum() - costs[ix].sum()) / max(hold[ix].sum() - orc[ix].sum(), 1e-9)
    pt = share(np.arange(len(costs))); bs = []
    for _ in range(B):
        ix = np.concatenate([rng.choice(ids[s], size=len(ids[s]), replace=True) for s in rng.choice(keys, size=len(keys), replace=True)])
        bs.append(share(ix))
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True); ap.add_argument("--truth", required=True); ap.add_argument("--responses", required=True)
    ap.add_argument("--out", default="results/llm/eval"); ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--kb", default=None)
    a = ap.parse_args(); cfg = Config()
    packets = {r["key"]: r for r in map(json.loads, open(a.packets))}
    truth = {r["key"]: r for r in map(json.loads, open(a.truth))}
    resp = defaultdict(list); metas = []
    for r in map(json.loads, open(a.responses)):
        resp[(r["key"], r["arm"])].append(r); metas.append(r["meta"])
    mock = any(m.get("mock") for m in metas)
    if mock:
        print("\n*** WARNING: responses come from a MOCK test double. These numbers test the pipeline only and are NOT language-model results. ***\n")
    kb = {c.id: c for c in load_kb(a.kb)}
    groups = defaultdict(list)
    for k, p in packets.items():
        if any(kk[0] == k for kk in resp):
            groups[(p["network"], p["K"], p["seed"])].append(k)
    inc_rows, resp_rows, prop_rows, conf = [], [], [], []
    for (net, K, seed), keys in groups.items():
        t0 = time.time(); p0 = packets[keys[0]]
        cfg.leak_frac_range = tuple(p0["leak_frac"]); st = build_stack(net, cfg, seed, K)
        incs = generate_incidents(st.plant.base, cfg, np.random.default_rng(seed + 2000), dict(zip(("LEAK", "PUMP_FAILURE", "DEMAND_SURGE"), p0["counts"])))
        for key in keys:
            rec = packets[key]; prep = prepare_incident(st, cfg, seed, rec["idx"], incs[rec["idx"]])
            if prep is None or packet_hash(prep.packet) != rec["packet_hash"]:
                print(f"  skip {key}: packet could not be reproduced (hash mismatch); was the code or seed changed?"); continue
            ref = reference_outcomes(prep, st, cfg); kind = truth[key]["kind"]
            hits = [kb[c] for c in rec["retrieved"] if c in kb]
            inc_rows.append(dict(key=key, seed=seed, kind=kind, hold=prep.hold.objective, oracle=ref["oracle"].objective, naive=ref["naive"].objective,
                                 search=ref["search"].objective, naive_unsafe=not ref["naive"].hard_safe, search_unsafe=not ref["search"].hard_safe,
                                 naive_harmful=ref["naive"].objective > prep.hold.objective * 1.05 + 5, search_harmful=ref["search"].objective > prep.hold.objective * 1.05 + 5,
                                 retrieval_hit=any(REL[kind] & set(c.tags) for c in hits), n_relevant_in_kb=sum(bool(REL[kind] & set(c.tags)) for c in kb.values())))
            for arm in ("direct", "rag"):
                for r in resp.get((key, arm), []):
                    parsed = parse_response(r["text"]) if r["ok"] else Parsed("failed", error="call failed: " + r.get("error", ""))
                    e = evaluate_response(prep, st, cfg, parsed)
                    for pr in e.pop("proposals"):
                        prop_rows.append(dict(key=key, arm=arm, **pr))
                    conf += [dict(arm=arm, twin_safe=t, plant_safe=pl) for t, pl in e.pop("twin_vs_plant")]
                    u = r["meta"].get("usage") or {}
                    resp_rows.append(dict(key=key, seed=seed, kind=kind, arm=arm, rep=r["rep"], call_ok=r["ok"], first_key=e["first_valid_key"],
                                          prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"), latency_s=r["meta"].get("latency_s"),
                                          provider=r["meta"].get("provider"), served_model=r["meta"].get("served_model"), **{k: v for k, v in e.items() if k != "first_valid_key"}))
        print(f"[{net} K={K} seed {seed}] evaluated {len(keys)} incidents in {time.time()-t0:.0f}s", flush=True)
    I, R, P = pd.DataFrame(inc_rows), pd.DataFrame(resp_rows), pd.DataFrame(prop_rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); I.to_csv(a.out + "_incidents.csv", index=False); R.to_csv(a.out + "_responses.csv", index=False); P.to_csv(a.out + "_proposals.csv", index=False)
    rng = np.random.default_rng(0); seeds = I.seed.values; hold, orc = I.hold.values, I.oracle.values
    def per_inc(col, arm):
        g = R[R.arm == arm].groupby("key")[col].mean(); return I.key.map(g).values.astype(float)
    pol = [("HOLD (do nothing)", hold, np.zeros(len(I)), np.zeros(len(I)), np.zeros(len(I))),
           ("naive playbook on top diagnosis", I.naive.values, I.naive_unsafe.values.astype(float), I.naive_harmful.values.astype(float), np.zeros(len(I))),
           ("exhaustive search + twin verification (no LLM)", I.search.values, I.search_unsafe.values.astype(float), I.search_harmful.values.astype(float), np.zeros(len(I)))]
    for arm, label in (("direct", "LLM (no retrieval)"), ("rag", "LLM + retrieval")):
        if (R.arm == arm).any():
            pol.append((f"{label}, validator only", per_inc("direct_cost", arm), per_inc("direct_unsafe", arm), per_inc("direct_harmful", arm), per_inc("direct_deferred", arm)))
            pol.append((f"{label} + twin verification", per_inc("verified_cost", arm), per_inc("verified_unsafe", arm), per_inc("verified_harmful", arm), per_inc("verified_is_hold", arm)))
    pol.append(("oracle (true incident known)", orc, np.zeros(len(I)), np.zeros(len(I)), np.zeros(len(I))))
    rows = []
    for name, c, uns, harm, defer in pol:
        c = np.where(np.isnan(c), hold, c); pt, lo, hi = boot_share(c, hold, orc, seeds, a.boot, rng)
        rows.append(dict(policy=name, cost_vs_hold_pct=round(100 * (c.sum() / hold.sum() - 1), 1), share_of_oracle_gain_pct=round(pt, 1), ci95=f"[{lo:.0f}, {hi:.0f}]",
                         worse_than_hold_pct=round(100 * np.nanmean(harm), 1), hard_violation_pct=round(100 * np.nanmean(uns), 1), no_action_pct=round(100 * np.nanmean(defer), 1)))
    S = pd.DataFrame(rows); S.to_csv(a.out + "_policies.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(f"{len(I)} incidents ({dict(I.kind.value_counts())}), {len(R)} responses ({'MOCK' if mock else 'real'})\n"); print(S.to_string(index=False))
    if len(R):
        q = []
        for arm, g in R.groupby("arm"):
            pg = P[P.arm == arm]; v = pg[pg.valid & ~pg.review]
            q.append(dict(arm=arm, responses=len(g), parse_ok_pct=round(100 * (g.parse_status == "ok").mean(), 1), recovered_pct=round(100 * (g.parse_status == "recovered").mean(), 1),
                          parse_failed_pct=round(100 * (g.parse_status == "failed").mean(), 1), escalation_pct=round(100 * g.escalate.mean(), 1),
                          proposals_per_response=round(g.n_proposed.mean(), 2), invalid_proposal_pct=round(100 * (pg.valid == False).sum() / max(len(pg), 1), 1) if len(pg) else np.nan,
                          unknown_id_pct=round(100 * pg.unknown_id.mean(), 1) if len(pg) else np.nan, bad_kind_pct=round(100 * pg.bad_kind.mean(), 1) if len(pg) else np.nan,
                          unsafe_of_valid_pct=round(100 * v.unsafe.mean(), 1) if len(v) else np.nan, harmful_of_valid_pct=round(100 * v.harmful.mean(), 1) if len(v) else np.nan,
                          beneficial_of_valid_pct=round(100 * v.beneficial.mean(), 1) if len(v) else np.nan))
        print("\nProposal quality (up to 3 proposals per response; 'valid' = passes the validator; unsafe/harmful judged on the TRUE plant):"); print(pd.DataFrame(q).to_string(index=False))
        C = pd.DataFrame(conf)
        if len(C):
            print("\nTwin verdict vs plant verdict on valid proposals (hard-safety):")
            print(f"  plant-unsafe proposals vetoed by the twin: {100*(~C[~C.plant_safe].twin_safe).mean():.0f}% of {int((~C.plant_safe).sum())}"
                  f" | plant-safe proposals vetoed anyway (unsafe under at least one belief with weight >= 10%): {100*(~C[C.plant_safe].twin_safe).mean():.0f}% of {int(C.plant_safe.sum())}")
        cons = R.groupby(["key", "arm"]).first_key.agg(lambda s: np.mean([x == y for i, x in enumerate(s) for y in list(s)[i + 1:]]) if len(s) > 1 else np.nan)
        print("\nrepeat consistency (same first action across repeats):", cons.groupby("arm").mean().round(2).to_dict())
        print("retrieval hit@%d (a relevant chunk among those shown): %.0f%% (chance with random chunks: ~%.0f%%)" % (len(packets[next(iter(packets))]["retrieved"]), 100 * I.retrieval_hit.mean(),
              100 * (1 - np.prod([(len(kb) - I.n_relevant_in_kb.mean() - j) / (len(kb) - j) for j in range(len(packets[next(iter(packets))]["retrieved"]))]))))
        tok = R.groupby("arm")[["prompt_tokens", "completion_tokens", "latency_s"]].mean().round(1); print("\nmean tokens / latency per call:\n", tok.to_string())
        print("served models / providers:", {f"{m} / {pr}": int(n) for (m, pr), n in R.assign(served_model=R.served_model.fillna("-"), provider=R.provider.fillna("-")).groupby(["served_model", "provider"]).size().items()})
        print("call failures:", int((~R.call_ok).sum()), "of", len(R))
    print("wrote", a.out + "_{policies,responses,proposals,incidents}.csv")


if __name__ == "__main__":
    main()
