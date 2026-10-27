"""Phase 2: call a model on prepared packets (stdlib only; resumable). Reads packets only, never ground truth.
Real run:  OPENROUTER_API_KEY=... python scripts/llm_run.py --packets P --backend openrouter --model qwen/qwen3.8-27b --provider SomeProvider --repeats 3
Dry run:   python scripts/llm_run.py --packets P --backend mock --mock-mode diagnosis_top1   (TEST DOUBLE: results are not language-model results)"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wdnv.llm import OpenRouterClient, OllamaClient, MockLLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True); ap.add_argument("--out", default=None)
    ap.add_argument("--backend", choices=["openrouter", "ollama", "mock"], required=True); ap.add_argument("--model", default=None)
    ap.add_argument("--provider", default=None, help="comma-separated OpenRouter provider order; pins the provider (no fallbacks)")
    ap.add_argument("--arms", nargs="+", default=["direct", "rag"]); ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.2); ap.add_argument("--max-tokens", type=int, default=1500); ap.add_argument("--base-seed", type=int, default=1234)
    ap.add_argument("--no-json-mode", action="store_true"); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--mock-mode", default="diagnosis_top1"); ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args()
    out = a.out or a.packets.replace("_packets.jsonl", f"_responses_{a.backend}.jsonl")
    if a.backend == "openrouter":
        cl = OpenRouterClient(a.model, provider_order=a.provider.split(",") if a.provider else None, json_mode=not a.no_json_mode)
    elif a.backend == "ollama":
        cl = OllamaClient(a.model, json_mode=not a.no_json_mode)
    else:
        cl = MockLLM(a.mock_mode); print("*** MOCK BACKEND: test double, not a language model ***")
    done = {}
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line); done[(r["key"], r["arm"], r["rep"])] = r["ok"]
    packets = [json.loads(l) for l in open(a.packets)]
    if a.limit:
        packets = packets[: a.limit]
    f = open(out, "a"); n_ok = n_fail = 0; t0 = time.time()
    for pk in packets:
        for arm in a.arms:
            for rep in range(a.repeats):
                k = (pk["key"], arm, rep)
                if k in done and (done[k] or not a.retry_failed):
                    continue
                pr = pk["prompts"][arm]
                resp = cl.generate(pr["system"], pr["user"], a.temperature, a.base_seed + rep, a.max_tokens)
                f.write(json.dumps(dict(key=pk["key"], arm=arm, rep=rep, seed=a.base_seed + rep, ok=resp.ok, error=resp.error, text=resp.text, meta=resp.meta)) + "\n"); f.flush()
                n_ok += resp.ok; n_fail += (not resp.ok)
                if a.sleep:
                    time.sleep(a.sleep)
        print(f"  {pk['key']}: ok {n_ok} failed {n_fail} ({time.time()-t0:.0f}s)", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
