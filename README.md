# wdnv: simulation-verified decision support for water-distribution incidents  (v0.6)

Research code for studying one question: **when an AI agent (e.g. an LLM) proposes operational actions for a
water-network incident, how often are those actions unsafe or wrong, and how much do simulation-based
verification and abstention (hand the decision to a human) fix?**

**v0.3** = v0.1 (simulation, incidents, actions, verification, telemetry) + a SCADA-synchronised twin, an
alarm-conditioned **diagnosis** layer with calibrated beliefs, decision **policies** that act on those beliefs, and a
first **closed-loop staged response** (probe -> observe -> re-diagnose -> isolate), and the **LLM layer**
(prompt packets, retrieval over a knowledge base, resumable model runner, tolerant parser, validator + twin verification, evaluation on the true plant).
The LLM layer is built and tested with mock models only: **no real language model has been run yet** (the authoring sandbox has no network).
There is still **no LLM / RAG layer**. Nothing here is a final paper result; read "Findings so far" first.

## Quick start
```bash
pip install -r requirements.txt
python -m pytest -q                              # 46 tests, ~20 s
python scripts/healthcheck.py Net3 ky6           # is a network usable? (any .inp path works too)
python scripts/smoke_experiment.py               # incidents -> candidate actions -> oracle vs twin, perfect diagnosis (~45 s)
python scripts/diagnosis_experiment.py --networks Net3 --sensors 10 25            # ~1 min  (KY6: --networks ky6 --sensors 20 54, ~5 min)
python scripts/end_to_end.py --networks Net3 --sensors 25 --cal 30 10 15 --test 30 8 12   # ~3 min
python scripts/closed_loop.py --sensors 25 --n-leak 14 --slice 0 5     # ~2.5 min per slice of 5 incidents; --merge to summarise
python scripts/hedging.py --sensors 25 --seeds 0 1 2                  # ~65 s per seed; then: python scripts/hedging_summary.py --sensors 25
```
LLM study (see `docs/LLM_STUDY_PROTOCOL.md`): `scripts/llm_prepare.py` -> `scripts/llm_run.py` (model calls, resumable) -> `scripts/llm_evaluate.py`.
(Run experiments in the foreground in pieces under ~4 minutes each; each writes CSVs to `results/`.)

## Layout
| module | role |
|---|---|
| `config.py` | all thresholds and weights in one dataclass (nothing hidden in code) |
| `network.py` | load + standardise a network (15-min steps, PDD), baseline run, `eligible` nodes, control handling |
| `incidents.py` | `LEAK` (burst on a pipe), `PUMP_FAILURE`, `DEMAND_SURGE`: injection + seeded random generator |
| `actions.py` | typed action space, **validation of hallucinated/unsafe commands**, scheduling, candidate enumeration |
| `metrics.py` | unmet demand, leak volume, pressure/tank constraints, objective |
| `verifier.py` | `evaluate` (one action on one model), `select` (safe-best / damage-limiting / review), imperfect `make_twin` |
| `telemetry.py` | noisy, partially missing sensor data derived from the true simulation; the twin's noise-free expectation |
| `sync.py` | SCADA-synchronised twin: runs on observed pump statuses; flags pumps whose status contradicts what the controls demand |
| `diagnosis.py` | signature library (leak on every pipe / pump trip / demand-surge zone), noise model, scoring, calibrated beliefs |
| `evaluation.py` | calibration error, credible sets, pipe-to-pipe hop distance, truth-to-hypothesis mapping |
| `policy.py` | candidate pool from beliefs, robust verification over the belief set, naive playbook baseline |
| `regions.py` | probe regions: node sets cut off from every source by closing a few boundary pipes |
| `closedloop.py` | staged probe -> observe -> re-simulate leading hypotheses under the action history -> isolate |
| `kb.py`, `data/knowledge_base.md` | 34-chunk operations knowledge base (with distractors) + TF-IDF retriever |
| `packet.py` | truth-free observation packet, system/user prompts, retrieval query |
| `llm.py` | OpenRouter / Ollama clients (stdlib, retries, provider pinning, never raise) and a MOCK test double |
| `proposal.py` | tolerant JSON parsing; failures counted (ok / recovered / failed) |
| `llm_pipeline.py` | prepare incidents, evaluate proposals (validator-only and twin-verified) on the plant |
| `pipeline.py` | one-call setup of plant, imperfect twin, library, sensors, synchronised twin, calibrated diagnoser |

## Design decisions (and the mistakes in v0.1-v0.9 they respond to)
* **Burst = split pipe + orifice to a sink, opened by a time control.** Isolation works only because the
  hydraulics cut the supply to the leak node; v0.9 set the leak to zero by hand. Test: leak flow after closing
  both sub-pipes is ~1e-4 L/s.
* **Fast engine, one run per scenario.** Everything runs in the EPANET engine (~30 ms Net3, ~110 ms KY6 per
  24 h). Actions are scheduled with controls at `t_action`; the trajectory before `t_action` is identical across
  candidates (tested), so candidates are compared fairly.
* **Operator overrides do not rewrite history.** Automatic controls on an asset are disabled only from the
  failure / action time onwards (`limit_controls_before`).
* **Constraints are checked where they can matter.** Only after `t_action`, only on `eligible` nodes
  (baseline min pressure >= 15 m), and "too high" means above the node's own daily baseline maximum. My first
  version compared with baseline at the same clock time, which flagged tank-cycle phase shifts as violations
  and made every action "unsafe"; the smoke test caught it.
* **Objective has physical units:** m3 of leaked water + m3 of unmet demand (+ a small per-operation tie-break).
  No hand-tuned score weights; `w_unmet`, `w_leak`, `ops_penalty_m3` are explicit and should be swept.
* **Solver failure is not a safety verdict.** `Outcome.status` separates `ok / invalid / sim_failed`.
* **Imperfect twin.** `make_twin` perturbs demand (global + per node) and pipe roughness; the plant stays exact.

## Diagnosis design (v0.6)
* **The alarm is an input** (latency 30-60 min, a parameter). Automatic detection was tried and does not work well here (see
  `NOTES_diagnosis.md`); the paper must not claim detection.
* Features are **post minus pre differences** around the alarm on pressure and tank-level residuals (cancels slow model mismatch),
  plus a **pump-status gap** channel. Pump-flow residuals are deliberately not used.
* The signature library is built on the *imperfect twin* (plant != twin), one simulation per hypothesis.
* Posterior = softmax(-chi2 / 2T) with temperatures fitted on a separate calibration set (NLL), then evaluated on a test set.
* Plants get **day-to-day demand variation** (`make_day`), so nothing can memorise one normal day.
* Isolation actions use **whole-pipe names**; closing a pipe closes both halves if it has been split by a (believed or true) burst,
  so a wrong belief produces a real, executable, harmful action.

## Validation evidence (measured, `results/validation_numbers.txt`)
* Orifice link vs analytic Cd*A*sqrt(2gP): max relative error 0.03 %.
* Network pressures vs WNTR's built-in leak model (WNTRSimulator), same burst: max difference 0.002 m.
* Baseline delivered / requested demand on eligible nodes: 1.000.
* 25 unit tests (incl. the synchronised twin reproducing a perfect plant: median pressure error 0.07 m, 95th pct 0.4 m): timing standardisation, demand accounting, pre-onset identity for all incident types,
  pump failure permanence, surge scaling, action validation, isolation-by-hydraulics, selection modes,
  solver-failure handling, twin and telemetry behaviour, KY6 smoke.
These are *verification* checks (the model does what it should, and agrees with WNTR). They are not validation
against real utility data.

## Findings so far (Net3 and KY6; single seed; synthetic incidents; NOT final)
**Diagnosis** (`results/diagnosis_*_summary.csv`; calibration on one incident set, test on another; alarm 45 min after onset):

| network | sensors | type acc. leak / surge / pump | type ECE | leak top-1 / top-10 | median hop error | 80 % credible set (size of all leak pipes) |
|---|---|---|---|---|---|---|
| Net3 | 10 | 49 % / 57 % / 95 % | 0.04 | 7 % / 41 % | 3 | 61 of 104 (covers truth 89 %) |
| Net3 | 25 | 59 % / 67 % / 100 % | 0.20 | 17 % / 51 % | 1 | 21 of 104 (covers truth 71 %) |
| KY6 | 20 | 62 % / 45 % / 73 % | 0.13 | 2 % / 8 % | 14 | 282 of 510 |
| KY6 | 54 | 62 % / 55 % / 87 % | 0.17 | 5 % / 22 % | 9 | 157 of 510 |

Pump trips are identified well when the controls leave the status unambiguous; leak *location* is the weak point and
gets worse with network size. Beliefs are usually vague, so decisions must be made under uncertainty.

**Decisions** (`results/e2e_net3_k25_*.csv`; Net3, 25 sensors, 14 qualifying leaks; pump trips and surges fall below the
0.2 %-of-daily-demand severity filter on Net3): knowing the true incident (oracle) would cut lost water + unmet demand by 81 % vs doing nothing.
The naive playbook on the top belief recovers 6 % of that gain, verification on the top belief 6 %, verification over the belief set 11 %;
no policy violated a hard constraint. The 'abstain' rows defer to an expert holding the *true* diagnosis after 2 h (84-87 % recovered)
and are an assumption, not a measured benefit.

**Closed-loop staged response** (`results/closedloop_K25_*.csv`; Net3, 25 sensors, the same 14 leaks for every policy; leak-only; one seed):

| policy | cost vs HOLD | share of oracle gain | worse than HOLD | hard-constraint violations |
|---|---|---|---|---|
| single-shot robust verification | -12.7 % | 15.6 % | 0 % | 0 % |
| closed loop, **no probing** (verify isolating the top-5 candidate pipes, hedged) | -23.4 % | **28.8 %** | 7 % | 0 % |
| closed loop with probing (value-of-information rule) | -16.2 % | 20.0 % | 29 % | 21 % |
| closed loop, forced probing | -15.4 % | 19.0 % | 36 % | 21 % |
| oracle (true pipe, immediate) | -81.1 % | 100 % | 0 % | 0 % |

The final top pipe was exactly right in only 3 of 14 incidents. *Caveat added after the 10-seed study below:* this is one seed and the
'no probing' variant is twin-verified selection among top-k isolations; unverified hedging actually hurts. **Probing did not pay**: it removed slightly more leak volume (31.0k vs 31.7k m3) but cut
off far more demand (4.7k vs 0.7k m3 unmet), consistent with the value-of-information estimate (a probe on a flat posterior resolves
< 10 % of the remaining uncertainty while its outage costs about as much as the leak flow). Feasibility: regions with <= 10 boundary valves
cover up to 36 of Net3's 104 candidate pipes but only up to 26 of KY6's 510, so blind probing cannot scale to KY6-sized networks.
The loop treats an incident as a leak when P(leak) >= 0.25 (calibrated type posteriors are nearly flat), so the type question is not
tested here. Bursts of 15-30 % of demand were not re-run after the corrections below.

**Hedging breadth, 10 seeds** (`scripts/hedging.py`, `scripts/hedging_summary.py`; `results/hedging_K*_*`). Isolate the top-k most likely
pipes at once (k = 1..6), unverified, or let the twin choose k (or do nothing) by expected outcome under the belief set. Every seed redraws
twin error, sensor placement, calibration, day-to-day demand and incidents; 14 Net3 leaks per seed; intervals are two-level bootstrap
(seeds, then incidents). 'Share' = share of the oracle's gain.

| 25 sensors (10 seeds, 140 incidents) | share of oracle gain [95 % CI] | worse than HOLD | hard violations | mean unmet demand |
|---|---|---|---|---|
| isolate top-1 (unverified) | 9 % [1, 21] | 11 % | 12 % | 127 m3 |
| isolate top-3 (unverified) | 1 % [-23, 22] | 40 % | 19 % | 657 m3 |
| isolate top-6 (unverified) | -41 % [-82, -9] | 59 % | 30 % | 1808 m3 |
| **twin-verified choice of k** (15 m threshold) | **27 % [10, 46]** | 16 % | 1.4 % | 150 m3 |

Paired difference, verified minus unverified top-1: +17.5 points [5.6, 30.0]. With **10 sensors** (6 seeds, 84 incidents) unverified top-1 is
*harmful* (-17 % [-61, 2]; cost +12.5 % vs doing nothing) and verification recovers only 5 % [-5, 16] (it mostly learns to do nothing; paired
difference +21.5 [-5, 71]). Diagnosis coverage (true pipe within the top-k): 25 sensors 11 % / 22 % / 29 % / 42 % for k = 1 / 3 / 6 / 10;
10 sensors 1 % / 10 % / 20 % / 30 %. **Between-seed spread is large** (25 sensors: top-1 coverage 0-29 %, share of gain from -2 % to +74 % per seed),
so any single-seed number, including my earlier ones, is unreliable. The verified policy is insensitive to the pressure-envelope
threshold between 15 and 20 m and slightly worse at 10 m (23 % vs 27 %). Net: *verification makes acting on uncertain locations safe and
roughly triples the benefit, but the benefit is capped by diagnosis quality (location is known well enough in only ~1 in 4 incidents).*

## LLM layer (status: pipeline complete, real-model results pending)
Design choices that respond to the flaws of earlier prototypes: the prompt never contains scenario labels or the true incident (a test checks the packet builder cannot
see it; packets are hash-checked at evaluation); the knowledge base is generic and includes 8 distractor chunks; invalid ids and kinds are *counted*, not silently
filtered; parse failures are counted as a metric; every call logs requested/served model, provider, usage, latency, seed and timestamp; the model is evaluated both
validator-only and with twin verification on the *same* responses; development and test seeds are separated in the protocol.
`results/llm_mock_example/` holds a **mock** dry run (6 Net3 leaks, chaos mock) showing the formats; it is not evidence about any language model.
One thing the dry run showed: the robust twin rule (safe under every belief with weight >= 10 %) vetoed 80-83 % of proposals that were safe on the true plant and led to
'no action' in 50-60 % of incidents, with the same gain as the unverified naive playbook; the veto rate must be reported next to the gain.

## Corrections made during this work (read these)
1. **Constraint flaw (fixed).** Customers *deliberately* cut off by an isolation were counted as pressure violations, so the oracle and the
   verifier refused beneficial isolations (e.g. -72 %) and fell back to doing nothing. Nodes with no path to a source are now excluded from the
   pressure constraints (the outage is charged as unmet demand; regression test added). Every experiment before this fix was wrong;
   e.g. the KY6 oracle gain on leaks was 42 % (median) and is 88 % after the fix; I had misread this as "isolating cuts customers off".
2. **Threshold choice (disclosed).** `hard_envelope_excess_m` was 10 m, an arbitrary early choice. An isolation that cut the cost by 92 % was
   rejected for exceeding it by 0.1 m. Oracle behaviour on the 14 Net3 leaks vs the threshold:

   | threshold (m) | 5 | 10 | 15 | 20 | 30 | 50 |
   |---|---|---|---|---|---|---|
   | incidents where the oracle isolates (of 14) | 12 | 12 | 13 | 13 | 13 | 13 |
   | oracle cost as % of HOLD | 30.2 | 27.5 | 18.9 | 18.9 | 18.9 | 18.9 |

   I set the default to 15 m (the start of the plateau) **after seeing this**; report results at several thresholds.
3. **Closed-loop type gate (fixed).** With P(leak) >= 0.5 required, probing was silently disabled (type posteriors are flat), so an earlier
   "forced probing" ablation did not probe. The gate is now a parameter (`--min-leak-prob`).

## Known limitations (read before trusting any number)
* Automatic detection is not solved (alarm is given). Diagnosis uses 10-100 random-ish pressure sensors; placement is not optimised.
* Closed-loop probing prototype is crude (probes = region closures only; leak-only; Net3 only; n = 14; one seed; no CIs). The hedging study has CIs but is also leak-only and Net3-only.
* No LLM/RAG, no baselines other than HOLD and exhaustive search.
* Candidate pool is a hand-built list (~10-20 actions). An open, parametric action space is needed for the LLM to matter.
* Replay-from-t=0 twin; the synchronised twin uses observed pump status but not observed tank levels as initial state.
* End-to-end results are one seed, Net3 only (KY6 end-to-end not run); small n; no confidence intervals yet.
* On Net3 only leaks are severe enough to matter; pump trips/surges are absorbed by the network (a property of this network).
* Two networks. POWER pumps (KY) cannot change speed; only on/off. Leaks are orifice links (no
  leak-end/repair logic tested yet). EPANET 2.2 PDA semantics, not WNTR's own solver.
* A few nodes have negative baseline pressure in both networks (excluded via `eligible`).

## Roadmap
1. **Raise the ceiling set by diagnosis:** sensor-placement strategies (not random-ish farthest-point), sharper localisation, KY6 and a third network,
   burst sizes; then revisit probing with outage-aware, faster-restoring probes. Report everything at several constraint thresholds with seed CIs.
2. Sensor-placement strategies (and zone-level isolation) as experimental factors; KY6 end-to-end; confidence intervals over seeds.
3. LLM/RAG layer as a *proposer* of candidate actions and diagnoses: independent knowledge base, no scenario labels in prompts, real
   models with fixed settings, logged provider/version, tolerant JSON parsing counted as a metric. Compare against the exhaustive
   search and the naive playbook.
4. Larger experiments: hundreds of procedural incidents, 3+ networks, plant/twin mismatch sweep, weight sensitivity,
   bootstrap CIs, paired tests, pre-registered hypotheses.
