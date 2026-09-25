"""Optional LLM layer.

The MVP runs fully offline: policy answers are extractive (the best
sentence from the retrieved chunk), which is always grounded. If
ANTHROPIC_API_KEY is set, `phrase_policy_answer` asks a small, fast model to
rephrase that grounded context into one natural spoken sentence in the
caller's language, and `classify_intent` handles utterances the rules miss.
Both are called only on the long tail, so spend stays small.
"""
from __future__ import annotations

import json
import os
import re

import httpx

from .rag.text import normalise

MODEL = os.environ.get("VOXARA_LLM_MODEL", "claude-haiku-4-5-20251001")
API = "https://api.anthropic.com/v1/messages"


def enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _call(system: str, user: str, max_tokens: int = 120, timeout: float = 4.0) -> str | None:
    try:
        r = httpx.post(API, timeout=timeout, headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": MODEL, "max_tokens": max_tokens, "system": system,
                  "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"]).strip()
    except Exception:
        return None     # never let the LLM take the call down; fall back to extractive


def extractive_answer(question: str, context: str) -> str:
    qt = set(normalise(question))
    best, best_s = "", -1.0
    for sent in re.split(r"(?<=[.?!])\s+", context):
        body = sent.split(". ", 1)[-1] if sent.count(". ") and len(sent.split(". ", 1)[0]) < 60 else sent
        s = len(qt & set(normalise(body))) / (len(set(normalise(body))) ** 0.5 or 1)
        if s > best_s:
            best, best_s = body, s
    return best.strip()


def phrase_policy_answer(question: str, context: str, lang: str, tone: str) -> str:
    if enabled():
        lang_name = "Hinglish (romanised Hindi mixed with English)" if lang.startswith("hi") else "Indian English"
        out = _call(
            "You are a delivery-call voice agent. Answer ONLY from the policy context. "
            "If the context does not answer the question, reply exactly UNKNOWN. "
            f"Reply in one short spoken sentence in {lang_name}, tone: {tone}. "
            "Never promise exact times, refunds or anything not in the context.",
            f"Policy context:\n{context}\n\nCustomer said: {question}")
        if out and out != "UNKNOWN":
            return out
    return extractive_answer(question, context)


INTENTS = ["reschedule", "refuse", "fake_attempt", "address_change", "cancel", "prepaid",
           "amount_query", "human", "refund", "delivery_time", "affirm", "deny", "unknown"]


def classify_intent(text: str) -> str | None:
    if not enabled():
        return None
    out = _call("Classify the delivery-call utterance into exactly one label from: "
                + ", ".join(INTENTS) + '. Reply as JSON {"intent": "..."}.', text, 30, 3.0)
    try:
        intent = json.loads(out or "{}").get("intent")
        return intent if intent in INTENTS else None
    except json.JSONDecodeError:
        return None
