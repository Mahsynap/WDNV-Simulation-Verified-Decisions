import json
import numpy as np
import pytest
import urllib.error
from wdnv.actions import Action, validate_action
from wdnv.kb import load_kb, TfidfRetriever
from wdnv.llm import MockLLM, OpenRouterClient, OllamaClient, build_openrouter_request, extract_packet, LLMResponse
from wdnv.proposal import parse_response
from wdnv.packet import render_prompts, retrieval_query, SYSTEM_PROMPT

PK = {"alarm_time_of_day": "05:00", "pumps": [{"id": "10", "type": "HEAD", "speed_adjustable": True, "observed_status": "on", "demanded_status": "on", "observed_flow_Ls": 100.0}],
      "tanks": [{"id": "1", "level_m": 5.0, "change_last_hour_m": -0.8, "min_level_m": 1, "max_level_m": 10}],
      "diagnosis": {"type_probabilities": {"LEAK": 0.5, "PUMP_FAILURE": 0.1, "DEMAND_SURGE": 0.3, "NONE": 0.1},
                    "top_leak_candidates": [{"pipe": "109", "probability": 0.2, "end_nodes": ["a", "b"], "adjacent_pipes": ["110"], "estimated_leak_Ls_if_leak": 30.0},
                                            {"pipe": "207", "probability": 0.1, "end_nodes": ["c", "d"], "adjacent_pipes": [], "estimated_leak_Ls_if_leak": 30.0}]}}


def test_parser_accepts_clean_fenced_and_prose_wrapped_json_and_counts_the_difference():
    obj = {"assessment": "x", "most_likely_incident": "LEAK", "actions": [{"kind": "CLOSE_PIPES", "params": {"pipes": ["109"]}}], "escalate_to_operator": False}
    t = json.dumps(obj)
    assert parse_response(t).status == "ok"
    assert parse_response("```json\n" + t + "\n```").status == "recovered"
    assert parse_response("Sure! Here you go:\n" + t + "\nAnything else?").status == "recovered"
    p = parse_response(t)
    assert p.actions[0].kind == "CLOSE_PIPES" and p.n_raw_actions == 1 and not p.escalate
    for bad in ("", "   ", "no json here", '{"actions": [', '{"actions": "close everything"}', "[1, 2, 3]"):
        assert parse_response(bad).status == "failed", bad


def test_parser_keeps_invalid_kinds_so_that_they_can_be_counted(net3, cfg):
    p = parse_response(json.dumps({"actions": [{"kind": "open_all_valves", "params": {}}, {"kind": "CLOSE_PIPES", "params": {"pipes": ["NOPE"]}}, "text"]}))
    assert [a.kind for a in p.actions] == ["OPEN_ALL_VALVES", "CLOSE_PIPES", "INVALID"]
    assert all(validate_action(net3.wn, a, cfg) for a in p.actions)
    assert parse_response(json.dumps({"actions": [], "escalate_to_operator": True})).escalate


def test_balanced_json_extraction_handles_braces_inside_strings():
    t = 'prefix {"actions": [{"kind": "HOLD", "params": {}, "rationale": "keep } this { text"}], "escalate_to_operator": false} suffix'
    p = parse_response(t)
    assert p.status == "recovered" and p.actions[0].kind == "HOLD"


def test_mock_chaos_produces_every_failure_mode_deterministically():
    user = "x <<<PACKET\n" + json.dumps(PK) + "\nPACKET>>> y"
    assert extract_packet(user)["alarm_time_of_day"] == "05:00"
    stats = {"ok": 0, "recovered": 0, "failed": 0}; bad_ids = 0
    for seed in range(3):
        m = MockLLM("chaos", seed=seed, p_bad_id=0.3, p_trunc=0.1, p_empty=0.05)
        for _ in range(150):
            r = m.generate("s", user)
            assert r.meta["mock"] is True
            p = parse_response(r.text); stats[p.status] += 1
            bad_ids += any(a.kind == "CLOSE_PIPES" and a.params["pipes"][0].startswith("P9") for a in p.actions)
    assert stats["ok"] > 0 and stats["recovered"] > 0 and stats["failed"] > 0 and bad_ids > 0
    r1 = MockLLM("chaos", seed=5).generate("s", user).text
    assert r1 == MockLLM("chaos", seed=5).generate("s", user).text


def test_packet_prompts_contain_no_ground_truth_markers_and_rag_only_adds_notes():
    chunks = TfidfRetriever(load_kb()).search(retrieval_query(PK), 3)
    s0, u0 = render_prompts(PK); s1, u1 = render_prompts(PK, [c for _, c in chunks])
    assert s0 == s1 and u1.startswith(u0.split("\nRecommend actions now")[0])
    assert "REFERENCE NOTES" in u1 and "REFERENCE NOTES" not in u0
    for forbidden in ("true_pipe", "ground truth", "incident_id", "LEA0", "PUM0", "SUR0"):
        assert forbidden not in u0 and forbidden not in SYSTEM_PROMPT


def test_retrieval_query_depends_only_on_the_packet_and_surfaces_relevant_guidance():
    pump_pk = json.loads(json.dumps(PK)); pump_pk["pumps"][0].update(observed_status="off", demanded_status="on")
    ret = TfidfRetriever(load_kb())
    ids = [c.id for _, c in ret.search(retrieval_query(pump_pk), 3)]
    assert "KB-07" in ids
    assert retrieval_query(PK) == retrieval_query(json.loads(json.dumps(PK)))


def test_kb_has_distractors_and_tags():
    kb = load_kb()
    tags = {t for c in kb for t in c.tags}
    assert len(kb) >= 30 and {"leak", "pump", "surge", "distractor", "uncertainty"} <= tags


def test_openrouter_request_construction_and_key_handling(monkeypatch):
    url, headers, body = build_openrouter_request("m/x", "sys", "usr", 0.2, 7, 500, True, ["P1"], "KEY", referer="https://e.org")
    assert url.endswith("/chat/completions") and headers["Authorization"] == "Bearer KEY" and headers["HTTP-Referer"] == "https://e.org"
    assert body["seed"] == 7 and body["response_format"] == {"type": "json_object"} and body["provider"] == {"order": ["P1"], "allow_fallbacks": False}
    assert body["messages"][0]["role"] == "system" and body["temperature"] == 0.2
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenRouterClient("m/x")


class _Resp:
    def __init__(self, payload): self.p = payload
    def read(self): return json.dumps(self.p).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_client_retries_on_429_then_succeeds_and_logs_provider_without_the_key():
    calls = {"n": 0}
    def opener(req, timeout=0):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "rate", {"Retry-After": "0"}, None)
        return _Resp({"choices": [{"message": {"content": '{"actions": []}'}, "finish_reason": "stop"}], "model": "m/x-2026", "provider": "Prov", "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "id": "gen-1"})
    c = OpenRouterClient("m/x", api_key="SECRET", opener=opener, sleep=lambda s: None, retries=4)
    r = c.generate("s", "u", seed=3)
    assert r.ok and calls["n"] == 3 and r.meta["provider"] == "Prov" and r.meta["served_model"] == "m/x-2026" and r.meta["attempts"] == 3
    assert "SECRET" not in json.dumps(r.meta)


def test_client_never_raises_on_persistent_failure_or_garbage():
    def boom(req, timeout=0): raise urllib.error.URLError("offline")
    r = OpenRouterClient("m/x", api_key="K", opener=boom, sleep=lambda s: None, retries=2).generate("s", "u")
    assert (not r.ok) and "URLError" in r.error
    class Garbage(_Resp):
        def read(self): return b"<html>not json</html>"
    r2 = OllamaClient("llama", opener=lambda req, timeout=0: Garbage({}), sleep=lambda s: None).generate("s", "u")
    assert not r2.ok and "bad JSON" in r2.error
    r3 = OpenRouterClient("m/x", api_key="K", opener=lambda req, timeout=0: _Resp({"error": {"message": "no credits"}}), sleep=lambda s: None).generate("s", "u")
    assert not r3.ok and "provider error" in r3.error
    r4 = OpenRouterClient("m/x", api_key="K", opener=lambda req, timeout=0: _Resp({"choices": [{"message": {"content": None}}]}), sleep=lambda s: None).generate("s", "u")
    assert r4.ok and r4.text == ""                                            # reasoning-only output: empty text, counted as a parse failure later
