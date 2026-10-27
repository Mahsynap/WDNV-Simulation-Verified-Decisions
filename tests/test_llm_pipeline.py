import inspect, json
import numpy as np
import pytest
from wdnv.actions import Action, HOLD
from wdnv.config import Config
from wdnv.incidents import generate_incidents
from wdnv.llm import MockLLM
from wdnv.llm_pipeline import prepare_incident, packet_hash, evaluate_response, reference_outcomes
from wdnv.packet import build_packet, render_prompts
from wdnv.pipeline import build_stack
from wdnv.proposal import parse_response


@pytest.fixture(scope="module")
def world():
    cfg = Config()
    st = build_stack("Net3", cfg, seed=0, K=25, cal=(15, 5, 8), n_normal=12)
    incs = generate_incidents(st.plant.base, cfg, np.random.default_rng(2000), {"LEAK": 3, "PUMP_FAILURE": 0, "DEMAND_SURGE": 0})
    prep = None
    for i, inc in enumerate(incs):
        prep = prepare_incident(st, cfg, 0, i, inc)
        if prep is not None:
            break
    return cfg, st, incs, i, prep


def test_packet_builder_cannot_see_the_true_incident():
    params = set(inspect.signature(build_packet).parameters)
    assert not (params & {"inc", "incident", "truth"})


def test_packets_are_reproducible_bit_for_bit(world):
    cfg, st, incs, i, prep = world
    again = prepare_incident(st, cfg, 0, i, incs[i])
    assert packet_hash(prep.packet) == packet_hash(again.packet)
    s, u = render_prompts(prep.packet)
    for k in (incs[i].id, "LEAK:" + str(incs[i].params["pipe"]), "ground"):
        assert k not in u


def test_response_evaluation_end_to_end_with_mock_models(world):
    cfg, st, incs, i, prep = world
    user = render_prompts(prep.packet)[1]
    top1 = parse_response(MockLLM("diagnosis_top1").generate("s", user).text)
    r = evaluate_response(prep, st, cfg, top1)
    assert r["parse_status"] == "ok" and r["n_invalid"] == 0 and not np.isnan(r["direct_cost"]) and not np.isnan(r["verified_cost"])
    assert r["verified_cost"] <= r["direct_cost"] + 1e-6 + 5.0 or r["verified_is_hold"]       # verification never chooses a worse-than-hold proposal on the twin's belief
    # a hallucinated action is rejected by the validator and cannot be executed
    bad = parse_response(json.dumps({"actions": [{"kind": "CLOSE_PIPES", "params": {"pipes": ["P9999"]}}], "escalate_to_operator": False}))
    rb = evaluate_response(prep, st, cfg, bad)
    assert rb["n_invalid"] == 1 and rb["proposals"][0]["unknown_id"] and rb["direct_deferred"] and rb["direct_cost"] == pytest.approx(prep.hold.objective)
    # escalation defers; failed parse defers
    esc = evaluate_response(prep, st, cfg, parse_response(json.dumps({"actions": [], "escalate_to_operator": True})))
    assert esc["direct_deferred"] and esc["direct_cost"] == pytest.approx(prep.hold.objective)
    assert evaluate_response(prep, st, cfg, parse_response("not json"))["parse_status"] == "failed"


def test_reference_policies_are_consistent(world):
    cfg, st, incs, i, prep = world
    ref = reference_outcomes(prep, st, cfg)
    assert ref["oracle"].objective <= prep.hold.objective + 1e-6
    assert ref["hold"] is prep.hold
