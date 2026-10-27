"""Tolerant parsing of model output into structured proposals. Parse failures are COUNTED, never silently repaired away."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field

from .actions import Action


@dataclass
class Parsed:
    status: str                     # ok | recovered | failed
    actions: list = field(default_factory=list)       # list[Action] in the model's order (invalid kinds kept)
    escalate: bool = False
    incident_guess: str = ""
    n_raw_actions: int = 0
    error: str = ""


def _balanced_json(text: str):
    """first balanced {...} object in text (string- and escape-aware), or None"""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def parse_response(text: str) -> Parsed:
    if not text or not text.strip():
        return Parsed("failed", error="empty response")
    obj, status = None, "ok"
    try:
        obj = json.loads(text)
    except Exception:
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
        for cand in (t, _balanced_json(t) or "", _balanced_json(text) or ""):
            try:
                obj = json.loads(cand); status = "recovered"; break
            except Exception:
                continue
    if not isinstance(obj, dict):
        return Parsed("failed", error="no JSON object found")
    raw = obj.get("actions", [])
    if not isinstance(raw, list):
        return Parsed("failed", error="'actions' is not a list")
    acts = [Action.from_dict(a) for a in raw[:10]]
    esc = bool(obj.get("escalate_to_operator", False)) or any(a.kind == "OPERATOR_REVIEW" for a in acts)
    return Parsed(status, acts, esc, str(obj.get("most_likely_incident", ""))[:20], len(raw))
