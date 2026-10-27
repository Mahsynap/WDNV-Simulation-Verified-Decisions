"""Phase 1: simulate incidents, run diagnosis, write prompt packets (what the model sees) and, separately, ground truth.
usage: python scripts/llm_prepare.py --network Net3 --sensors 25 --seeds 0 1 --counts 12 0 0 --out results/llm/net3"""
import argparse, json, os, sys, time, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("wntr").setLevel(logging.ERROR)
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wdnv.config import Config
from wdnv.incidents import generate_incidents
from wdnv.pipeline import build_stack
from wdnv.llm_pipeline import prepare_incident, packet_hash
from wdnv.kb import load_kb, TfidfRetriever
from wdnv.packet import render_prompts, retrieval_query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="Net3"); ap.add_argument("--sensors", type=int, default=25)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0]); ap.add_argument("--counts", nargs=3, type=int, default=[12, 0, 0], help="leak pump surge")
    ap.add_argument("--leak-frac", nargs=2, type=float, default=[0.03, 0.12]); ap.add_argument("--kb", default=None); ap.add_argument("--top-chunks", type=int, default=3)
    ap.add_argument("--out", default="results/llm/run")
    a = ap.parse_args(); cfg = Config(); cfg.leak_frac_range = tuple(a.leak_frac)
    chunks = load_kb(a.kb); ret = TfidfRetriever(chunks)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fp, ft = open(a.out + "_packets.jsonl", "a"), open(a.out + "_truth.jsonl", "a")
    for seed in a.seeds:
        t0 = time.time(); st = build_stack(a.network, cfg, seed, a.sensors)
        incs = generate_incidents(st.plant.base, cfg, np.random.default_rng(seed + 2000), dict(zip(("LEAK", "PUMP_FAILURE", "DEMAND_SURGE"), a.counts)))
        n = 0
        for idx, inc in enumerate(incs):
            p = prepare_incident(st, cfg, seed, idx, inc)
            if p is None:
                continue
            key = f"{a.network}|K{a.sensors}|s{seed}|i{idx}"
            hits = ret.search(retrieval_query(p.packet), a.top_chunks)
            s0, u0 = render_prompts(p.packet); s1, u1 = render_prompts(p.packet, [c for _, c in hits])
            fp.write(json.dumps(dict(key=key, network=a.network, K=a.sensors, seed=seed, idx=idx, counts=a.counts, leak_frac=a.leak_frac, packet_hash=packet_hash(p.packet), packet=p.packet,
                                     retrieved=[c.id for _, c in hits], prompts=dict(direct=dict(system=s0, user=u0), rag=dict(system=s1, user=u1)))) + "\n")
            ft.write(json.dumps(dict(key=key, kind=inc.kind, t_start=inc.t_start, pipe=inc.params.get("pipe"), pump=inc.params.get("pump"),
                                     hold_cost=p.hold.objective)) + "\n"); n += 1
        fp.flush(); ft.flush()
        print(f"[seed {seed}] {n} packets in {time.time()-t0:.0f}s", flush=True)
    fp.close(); ft.close()


if __name__ == "__main__":
    main()
