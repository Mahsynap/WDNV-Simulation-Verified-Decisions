"""LLM clients. Standard library only. Never raise on provider errors: return LLMResponse(ok=False) so runs are resumable.
OpenRouter / Ollama code paths could NOT be exercised from the authoring sandbox (no network); they are covered by offline unit
tests of request construction and retry logic only. Check the provider documentation before the first real run."""
from __future__ import annotations
import json, os, random, re, time, urllib.error, urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

PACKET_OPEN, PACKET_CLOSE = "<<<PACKET", "PACKET>>>"


@dataclass
class LLMResponse:
    text: str
    ok: bool
    error: str = ""
    meta: dict = field(default_factory=dict)


class LLMClient:
    name = "base"

    def generate(self, system: str, user: str, temperature: float = 0.2, seed: int | None = None, max_tokens: int = 1500) -> LLMResponse:
        raise NotImplementedError


def build_openrouter_request(model, system, user, temperature, seed, max_tokens, json_mode, provider_order, api_key,
                             referer=None, title="wdnv"):
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature, "max_tokens": max_tokens}
    if seed is not None:
        body["seed"] = int(seed)
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if provider_order:                                         # pin the provider: different providers can serve different quantisations
        body["provider"] = {"order": list(provider_order), "allow_fallbacks": False}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "X-Title": title}
    if referer:
        headers["HTTP-Referer"] = referer
    return "https://openrouter.ai/api/v1/chat/completions", headers, body


class _HttpClient(LLMClient):
    def __init__(self, timeout=120.0, retries=4, backoff=2.0, sleep=time.sleep, opener=urllib.request.urlopen):
        self.timeout, self.retries, self.backoff, self.sleep, self.opener = timeout, retries, backoff, sleep, opener

    def _post(self, url, headers, body):
        """returns (data dict | None, error str, attempts, latency)"""
        t0, err = time.time(), ""
        for attempt in range(1, self.retries + 2):
            try:
                req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
                with self.opener(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8")), "", attempt, time.time() - t0
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}"
                if e.code in (408, 409, 429) or e.code >= 500:
                    wait = self.backoff ** attempt + random.random()
                    try:
                        wait = max(wait, float(e.headers.get("Retry-After", 0)))
                    except Exception:
                        pass
                    if attempt <= self.retries:
                        self.sleep(wait); continue
                return None, err, attempt, time.time() - t0
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                err = f"{type(e).__name__}: {str(e)[:80]}"
                if attempt <= self.retries:
                    self.sleep(self.backoff ** attempt + random.random()); continue
                return None, err, attempt, time.time() - t0
            except (json.JSONDecodeError, ValueError) as e:
                return None, f"bad JSON from provider: {str(e)[:60]}", attempt, time.time() - t0
        return None, err, self.retries + 1, time.time() - t0


class OpenRouterClient(_HttpClient):
    name = "openrouter"

    def __init__(self, model, api_key=None, provider_order=None, json_mode=True, referer=None, title="wdnv", **kw):
        super().__init__(**kw)
        self.model, self.provider_order, self.json_mode, self.referer, self.title = model, provider_order, json_mode, referer, title
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("set OPENROUTER_API_KEY (never put the key in a file that is shared)")

    def generate(self, system, user, temperature=0.2, seed=None, max_tokens=1500) -> LLMResponse:
        url, headers, body = build_openrouter_request(self.model, system, user, temperature, seed, max_tokens, self.json_mode,
                                                      self.provider_order, self.api_key, self.referer, self.title)
        data, err, attempts, lat = self._post(url, headers, body)
        meta = {"backend": "openrouter", "requested_model": self.model, "temperature": temperature, "seed": seed, "json_mode": self.json_mode,
                "provider_pinned": list(self.provider_order) if self.provider_order else None, "attempts": attempts, "latency_s": round(lat, 3),
                "timestamp": datetime.now(timezone.utc).isoformat()}
        if data is None:
            return LLMResponse("", False, err, meta)
        if isinstance(data.get("error"), dict):
            return LLMResponse("", False, f"provider error: {str(data['error'])[:120]}", meta)
        try:
            ch = data["choices"][0]
            text = ch["message"].get("content") or ""
            meta.update(finish_reason=ch.get("finish_reason"))
        except (KeyError, IndexError, TypeError, AttributeError):
            return LLMResponse("", False, "malformed response structure", meta)
        meta.update(served_model=data.get("model"), provider=data.get("provider"), usage=data.get("usage"), generation_id=data.get("id"))
        return LLMResponse(text, True, "", meta)


class OllamaClient(_HttpClient):
    name = "ollama"

    def __init__(self, model, host="http://localhost:11434", json_mode=True, **kw):
        super().__init__(**kw); self.model, self.host, self.json_mode = model, host.rstrip("/"), json_mode

    def generate(self, system, user, temperature=0.2, seed=None, max_tokens=1500) -> LLMResponse:
        body = {"model": self.model, "stream": False, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": temperature, "num_predict": max_tokens, **({"seed": int(seed)} if seed is not None else {})}}
        if self.json_mode:
            body["format"] = "json"
        data, err, attempts, lat = self._post(self.host + "/api/chat", {"Content-Type": "application/json"}, body)
        meta = {"backend": "ollama", "requested_model": self.model, "temperature": temperature, "seed": seed, "attempts": attempts,
                "latency_s": round(lat, 3), "timestamp": datetime.now(timezone.utc).isoformat()}
        if data is None:
            return LLMResponse("", False, err, meta)
        meta.update(usage={"prompt_tokens": data.get("prompt_eval_count"), "completion_tokens": data.get("eval_count")}, served_model=data.get("model"))
        return LLMResponse((data.get("message") or {}).get("content", ""), True, "", meta)


def extract_packet(user: str) -> dict:
    a, b = user.index(PACKET_OPEN) + len(PACKET_OPEN), user.index(PACKET_CLOSE)
    return json.loads(user[a:b])


class MockLLM(LLMClient):
    """TEST DOUBLE ONLY. Reads the packet and answers with a fixed heuristic; 'chaos' injects format and ID faults.
    Results obtained with this class must never be reported as language-model results."""
    name = "mock"

    def __init__(self, mode="diagnosis_top1", seed=0, p_bad_id=0.15, p_bad_kind=0.1, p_fence=0.15, p_prose=0.1, p_trunc=0.05, p_empty=0.03):
        self.mode, self.rng = mode, random.Random(seed)
        self.p = dict(bad_id=p_bad_id, bad_kind=p_bad_kind, fence=p_fence, prose=p_prose, trunc=p_trunc, empty=p_empty)

    def _actions(self, pk):
        cands = pk["diagnosis"]["top_leak_candidates"]; tp = pk["diagnosis"]["type_probabilities"]
        dead = [p for p in pk["pumps"] if p["demanded_status"] == "on" and p["observed_status"] == "off"]
        if dead:
            others = [p["id"] for p in pk["pumps"] if p["id"] != dead[0]["id"]]
            return [{"kind": "SET_PUMP_STATUS", "params": {"pump": others[0], "status": "OPEN"}, "rationale": "mock", "confidence": 0.8}] if others else [{"kind": "HOLD", "params": {}}]
        if self.mode == "cautious" and (not cands or cands[0]["probability"] < 0.3):
            return []
        if tp.get("DEMAND_SURGE", 0) > max(tp.get("LEAK", 0), 0.5) or not cands:
            return [{"kind": "HOLD", "params": {}, "rationale": "mock", "confidence": 0.5}]
        k = 3 if self.mode == "hedge3" else 1
        return [{"kind": "CLOSE_PIPES", "params": {"pipes": [c["pipe"] for c in cands[:k]]}, "rationale": "mock", "confidence": cands[0]["probability"]}]

    def generate(self, system, user, temperature=0.2, seed=None, max_tokens=1500) -> LLMResponse:
        pk = extract_packet(user)
        acts = self._actions(pk)
        if self.mode == "chaos":
            if acts and acts[0]["kind"] == "CLOSE_PIPES" and self.rng.random() < self.p["bad_id"]:
                acts[0]["params"]["pipes"][0] = "P" + str(self.rng.randint(9000, 9999))
            if acts and self.rng.random() < self.p["bad_kind"]:
                acts.insert(0, {"kind": "OPEN_ALL_VALVES", "params": {}, "rationale": "mock", "confidence": 0.9})
        obj = {"assessment": "mock assessment", "most_likely_incident": "UNCERTAIN", "actions": acts, "escalate_to_operator": not acts}
        text = json.dumps(obj)
        if self.mode == "chaos":
            r = self.rng.random()
            if r < self.p["empty"]:
                text = ""
            elif r < self.p["empty"] + self.p["trunc"]:
                text = text[: max(5, len(text) // 2)]
            elif r < self.p["empty"] + self.p["trunc"] + self.p["fence"]:
                text = "```json\n" + text + "\n```"
            elif r < self.p["empty"] + self.p["trunc"] + self.p["fence"] + self.p["prose"]:
                text = "Here is my recommendation:\n" + text + "\nLet me know if you need more detail."
        return LLMResponse(text, True, "", {"backend": "mock", "mock": True, "mode": self.mode, "requested_model": f"mock-{self.mode}",
                                            "served_model": f"mock-{self.mode}", "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": len(text) // 4},
                                            "latency_s": 0.0, "temperature": temperature, "seed": seed})
