"""Voice-optimised RAG pipeline.

What makes it "optimised" for a live phone call rather than a chatbot:

1. Route first. Order facts (status, amount, pincode, NDR reason) come from
   the OMS as structured data and are never retrieved from text; RAG is only
   for policy. That removes the most common hallucination source.
2. Intent-aware query rewrite. The NLU intent adds policy vocabulary to the
   caller's words ("kal aana" -> + "reattempt tomorrow"), which lifts recall
   on short, noisy ASR transcripts.
3. Metadata filtering by workflow before scoring, so an NDR call never pulls
   COD-prepaid rules into context.
4. Hybrid BM25 + dense retrieval fused with reciprocal-rank fusion, then a
   cheap lexical rerank. No cross-encoder: it costs 50-150 ms per call.
5. A confidence gate. Below threshold the agent does not answer from policy;
   it says it will check and hands off. "I don't know" beats a wrong rule.
6. A tight context budget (default ~600 chars, top 2-3 chunks) because TTS
   reads everything aloud and the LLM's latency grows with context.
7. An LRU cache keyed on (workflow, intent, normalised query). Delivery
   calls repeat the same few questions thousands of times a day.
8. Every answer carries chunk ids as citations, which go to the audit log.
"""
from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import load_kb
from .index import HybridIndex
from .text import normalise

INTENT_EXPANSION = {
    "reschedule": "reattempt delivery date slot schedule",
    "refuse": "refusal reason RTO return do not want",
    "fake_attempt": "fake attempt courier never came complaint priority reattempt",
    "address_change": "address incomplete landmark pincode change alternate phone",
    "cancel": "cancellation cancel order no fee before dispatch",
    "prepaid": "prepaid UPI link discount switching",
    "amount_query": "amount disputes price taxes COD fee",
    "human": "escalation human handoff transfer callback",
    "refund": "refund support team",
    "delivery_time": "delivery slot time guarantee courier",
    "recording": "recording consent objects",
    "do_not_call": "do not call suppressed attempts",
}

# Pure intents answered by the workflow engine need no policy lookup.
NO_RETRIEVAL_INTENTS = {"affirm", "deny", "greet", "thanks", "provide_date", "provide_slot"}


@dataclass
class RagResult:
    context: str
    citations: list[str]
    confidence: float
    latency_ms: float
    cached: bool = False
    hits: list[dict] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return self.confidence >= RagPipeline.CONFIDENCE_THRESHOLD


class RagPipeline:
    CONFIDENCE_THRESHOLD = 0.35

    def __init__(self, kb_dir: str | Path, k: int = 3, budget_chars: int = 600,
                 cache_size: int = 512, embed_fn=None):
        self.chunks = load_kb(kb_dir)
        self.index = HybridIndex(self.chunks, embed_fn=embed_fn)
        self.k, self.budget = k, budget_chars
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self.stats = {"queries": 0, "cache_hits": 0, "skipped": 0, "low_confidence": 0}

    _QUESTION = re.compile(r"\?|^(can|could|will|is|are|do|does|kya|kab|kaise)\b", re.I)

    def needs_retrieval(self, intent: str | None, query: str = "") -> bool:
        # "kal" is a slot answer; "can you deliver today?" is a policy question.
        return intent not in NO_RETRIEVAL_INTENTS or bool(self._QUESTION.search(query.strip()))

    def rewrite(self, query: str, intent: str | None) -> str:
        return f"{query} {INTENT_EXPANSION.get(intent or '', '')}".strip()

    def retrieve(self, query: str, intent: str | None = None, workflow: str | None = None,
                 lang: str | None = None) -> RagResult:
        t0 = time.perf_counter()
        self.stats["queries"] += 1
        if not self.needs_retrieval(intent, query):
            self.stats["skipped"] += 1
            return RagResult("", [], 1.0, (time.perf_counter() - t0) * 1000)

        key = (workflow, intent, " ".join(sorted(set(normalise(query)))))
        if key in self._cache:
            self._cache.move_to_end(key)
            self.stats["cache_hits"] += 1
            r = self._cache[key]
            return RagResult(r.context, r.citations, r.confidence,
                             (time.perf_counter() - t0) * 1000, True, r.hits)

        q = self.rewrite(query, intent)
        hits = self.index.search(q, k=self.k * 2, workflow=workflow, lang=lang)
        hits = self._rerank(q, hits)[: self.k]

        context, cites, used = [], [], 0
        for chunk, _, _ in hits:
            body = chunk.meta.get("body", chunk.text)   # titles help retrieval, not speech
            if used + len(body) > self.budget and context:
                break
            context.append(body)
            cites.append(chunk.id)
            used += len(body)

        conf = self._confidence(q, hits)
        if conf < self.CONFIDENCE_THRESHOLD:
            self.stats["low_confidence"] += 1
        res = RagResult("\n".join(context), cites, conf, (time.perf_counter() - t0) * 1000,
                        hits=[{"id": c.id, "score": round(s, 4), **{k: round(v, 3) for k, v in d.items()}}
                              for c, s, d in hits])
        self._cache[key] = res
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return res

    @staticmethod
    def _rerank(q: str, hits):
        qt = set(normalise(q))

        def score(h):
            ct = set(normalise(h[0].text))
            overlap = len(qt & ct) / (len(qt) or 1)
            return h[1] + 0.02 * overlap
        return sorted(hits, key=score, reverse=True)

    @staticmethod
    def _confidence(q: str, hits) -> float:
        """Share of the caller's meaningful terms the best chunk covers,
        blended with dense similarity. Cheap, monotone, easy to tune."""
        if not hits:
            return 0.0
        qt = set(normalise(q))
        best = hits[0]
        matched = len(qt & set(normalise(best[0].text)))
        coverage = matched / (len(qt) or 1)
        dense = max(0.0, best[2].get("dense", 0.0))
        conf = 0.6 * coverage + 0.4 * dense
        if matched < 2:          # one shared word ("PM" in "9 PM") is not evidence
            conf *= 0.5
        return round(conf, 3)
