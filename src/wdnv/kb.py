"""Knowledge base + a small TF-IDF retriever (numpy only)."""
from __future__ import annotations
import math, os, re
from collections import Counter
from dataclasses import dataclass, field
import numpy as np

_STOP = set("a an the and or of to in on for with is are be by as at it its this that from when if not no can may must should will do does than then so such any all each more most".split())
_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "knowledge_base.md")


@dataclass
class Chunk:
    id: str
    title: str
    text: str
    tags: list = field(default_factory=list)


def tokens(s: str) -> list:
    return [t for t in re.findall(r"[a-z]{3,}", s.lower()) if t not in _STOP]


def load_kb(path: str | None = None) -> list:
    path = path or _DEFAULT
    chunks, cur = [], None
    for line in open(path, encoding="utf-8"):
        if line.startswith("## "):
            if cur:
                chunks.append(cur)
            parts = [p.strip() for p in line[3:].split("|")]
            tags = [t.strip() for t in parts[1].replace("tags:", "").split(",")] if len(parts) > 1 else []
            cur = Chunk(parts[0], parts[2] if len(parts) > 2 else parts[0], "", tags)
        elif cur is not None and not line.startswith("#"):
            cur.text += line.strip() + " "
    if cur:
        chunks.append(cur)
    for c in chunks:
        c.text = c.text.strip()
    return chunks


class TfidfRetriever:
    def __init__(self, chunks: list):
        self.chunks = chunks
        docs = [tokens(c.title + " " + c.text) for c in chunks]
        self.vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
        df = Counter(w for d in docs for w in set(d))
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1.0 for w in self.vocab])
        self.M = np.vstack([self._vec(d) for d in docs])

    def _vec(self, toks: list) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for w, c in Counter(toks).items():
            if w in self.vocab:
                v[self.vocab[w]] = (1 + math.log(c)) * self.idf[self.vocab[w]]
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 0 else v

    def search(self, query: str, k: int = 3) -> list:
        sims = self.M @ self._vec(tokens(query))
        order = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.chunks[i]) for i in order]
