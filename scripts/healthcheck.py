"""Health check for candidate networks: does the network run, and is its normal operation sane?
usage: python scripts/healthcheck.py Net3 ky6 path/to/other.inp"""
import sys, time, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("wntr").setLevel(logging.ERROR)
import numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "src"))
from wdnv.config import Config
from wdnv.network import load_network, compute_baseline, requested_demand

cfg = Config()
rows = []
for name in sys.argv[1:]:
    try:
        wn = load_network(name, cfg)
        t0 = time.time(); b = compute_baseline(wn, cfg); dt = time.time() - t0
        P = b.pressure.values
        q = b.results.link["flowrate"]
        req = requested_demand(wn, b.times)[b.junctions].values.sum()
        dl = b.results.node["demand"][b.junctions].values.sum()
        issues = []
        if (P < 0).any(): issues.append(f"{(P.min(axis=0) < 0).sum()} nodes with negative pressure")
        if len(b.eligible) < 0.8 * len(b.junctions): issues.append(f"only {len(b.eligible)}/{len(b.junctions)} nodes eligible")
        idle = [p for p in wn.pump_name_list if (q[p].abs() < 1e-6).all()]
        if idle: issues.append(f"pumps never running: {idle}")
        if dl / req < 0.98: issues.append(f"baseline delivers only {dl/req:.1%} of requested demand")
        ptypes = sorted({wn.get_link(p).pump_type for p in wn.pump_name_list})
        rows.append((name, wn.num_junctions, wn.num_pipes, wn.num_pumps, "/".join(ptypes) or "-", wn.num_tanks,
                     len(list(wn.controls())), len(b.eligible), f"{P.min():.1f}", f"{np.median(P):.1f}", f"{P.max():.1f}", f"{dt:.2f}s", "; ".join(issues) or "none"))
    except Exception as e:
        rows.append((name, "FAIL", type(e).__name__ + ": " + str(e)[:60]) + ("",) * 10)
hdr = ("network", "junctions", "pipes", "pumps", "pump types", "tanks", "controls", "eligible", "P min", "P med", "P max", "run", "issues")
w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
for r in [hdr] + rows:
    print(" | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
