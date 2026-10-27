# Diagnosis: what was tried and why it changed (keep for the paper's limitations section)

Goal: detect and diagnose incidents from noisy, partial telemetry using an imperfect twin.

## Automatic detection did not work well (all on Net3 / KY6, 10-100 pressure sensors, leaks 3-30 % of mean demand)
| variant | outcome |
|---|---|
| per-channel offset + max abs z | threshold ~9 sigma; ~8 % of leaks detected |
| demand-gain correction from first 2 h | worse (gain not identifiable at night) |
| time-varying noise floor | modest: Net3 leaks 33 -> 47 %, KY6 0 -> 30 % |
| step / change-point statistic (post - pre windows) | Net3 ~47 % with 1.6 h delay; KY6 3-10 % |
| pooled pressure-drop statistic | worse |
| more sensors | worse: threshold of a max statistic grows with sensor count |
| schedule-driven pumps (shortcut) | introduced an artifact (115-170 m pressure jump when KY6 tanks fill; removed) |
| synchronised twin on emulated controls | removed some jitter, thresholds still high |

Root causes found: (1) level-controlled pumps switch at different moments in plant and twin; (2) a flat pump curve (Net3 pump 335)
turns a +-10 % roughness error into 844 vs 1480 L/s of flow; (3) leak signals at sparse sensors are small vs model mismatch;
(4) a threshold crossed *between* samples makes sampled tank levels unable to reveal the pump's hysteresis state.

## Resulting design
Alarm given; (post - pre) feature differences; pump-flow residuals dropped; synchronised twin driven by observed pump statuses;
status gap only where the controls make the demanded status unambiguous (outside hysteresis bands); empirical noise floor from normal
days; temperatures fitted on a separate incident set.

## Bugs found by tests/experiments (fixed)
* constraint compared with baseline at the same clock time -> tank-cycle phase shifts flagged as violations (v0.1 smoke test)
* controls on a failed/overridden pump removed for the whole day instead of from the event time on
* status gap only existed for controlled links -> an always-on pump trip (KY6 Pump-1) was invisible (pump accuracy 53 % -> 73-87 %)
* half-pipe isolation actions could not be executed when the believed leak pipe was wrong (now whole-pipe actions)

## Closed-loop prototype notes
* Probe value = (bits a probe can resolve / bits still unknown) x estimated leak flow x hours remaining - outage demand x stage hours.
  An earlier ad-hoc value (k_info x 4m(1-m) x q_leak - demand) never triggered probes.
* Hypothesis update re-simulates the top-M leak hypotheses on the synchronised twin under the *full action history*; likelihood = chi2 of the
  post-minus-pre residual features around the probe (noise floor fitted on normal days). A noise-free test confirms the true hypothesis
  explains the telemetry >5x better than a wrong one after a probe that cuts off the true pipe.
* Hedging (closing the two or three most likely pipes together, checked by the twin) is cheap and was responsible for most of the gain.

## Evaluation bugs found late (all fixed; see README "Corrections")
* deliberately cut-off nodes counted as pressure violations (made the oracle refuse good isolations)
* knife-edge envelope threshold (10 m) chosen early; sensitivity table in README
* closed-loop type gate (P(leak) >= 0.5) silently disabled probing

## Hedging study correction
The v0.4 statement "hedging is what helped" came from one lucky seed (seed 0: 21 % top-1 coverage). Over 10 seeds unverified wide hedging
is harmful (top-6: -41 % of oracle gain); only twin-verified selection helps (+27 % [10, 46]).
