"""Tokenisation and Hinglish normalisation shared by retrieval and NLU.

Indian callers mix Hindi and English, often romanised with inconsistent
spelling ("kal", "kl", "kall"). We map common variants onto one canonical
English token so a Hinglish query can match an English policy.
"""
from __future__ import annotations

import re

STOP = set("""a an the is are was were be been to of in on at for and or but if
then so it this that these those with as by from i me my we our you your he she
they them his her its do does did can could will would shall should may might
must not no yes please ji hai hain ho tha thi the ka ki ke ko se me mein par
what when where how why which who kya kab kaise kyun kaun hi bhi aap mera mujhe main karna kar do
""".split())

# Romanised Hindi / Hinglish -> canonical English token.
HINGLISH = {
    "kal": "tomorrow", "kl": "tomorrow", "kall": "tomorrow",
    "parso": "day_after_tomorrow", "parson": "day_after_tomorrow",
    "aaj": "today", "aj": "today", "abhi": "now",
    "subah": "morning", "dopahar": "afternoon", "dopehar": "afternoon",
    "shaam": "evening", "sham": "evening", "raat": "night",
    "pata": "address", "address": "address", "ghar": "house",
    "paisa": "amount", "paise": "amount", "rupaye": "amount", "rupees": "amount",
    "cancel": "cancel", "radd": "cancel",
    "nahi": "no", "nahin": "no", "nai": "no", "mat": "no",
    "haan": "yes", "ha": "yes", "haa": "yes", "hanji": "yes", "theek": "ok", "thik": "ok",
    "delivery": "delivery", "parcel": "order", "saman": "order", "samaan": "order",
    "boy": "courier", "courier": "courier", "wala": "courier",
    "phone": "phone", "call": "call", "number": "phone",
    "badalna": "change", "badal": "change", "change": "change",
    "chahiye": "want", "chaiye": "want",
    "insaan": "human", "aadmi": "human", "banda": "human", "agent": "human",
    "refund": "refund", "wapas": "return", "return": "return",
    "landmark": "landmark", "mandir": "temple", "school": "school",
    "online": "prepaid", "upi": "prepaid",
}

# English synonyms folded to one term so paraphrases match policy wording.
SYNONYMS = {
    "reschedule": "reattempt", "redeliver": "reattempt", "retry": "reattempt",
    "attempt": "reattempt", "again": "reattempt",
    "refused": "refusal", "refuse": "refusal", "reject": "refusal", "rejected": "refusal",
    "rto": "return", "returned": "return",
    "cod": "cod", "cash": "cod",
    "agent": "human", "person": "human", "manager": "human", "supervisor": "human",
    "slot": "slot", "time": "slot", "timing": "slot",
    "pincode": "pincode", "pin": "pincode", "zip": "pincode",
    "record": "recording", "recorded": "recording",
    "angry": "angry", "upset": "angry",
    "fake": "fake", "never": "fake", "didnt": "fake",
}

_TOKEN = re.compile(r"[a-z0-9ऀ-ॿ]+")


def normalise(text: str) -> list[str]:
    toks = []
    text = re.sub(r"\bhands?[\s-]+off", "handoff", text.lower())
    text = re.sub(r"\bcall(?:ed|s)?\s+(?:me\s+|you\s+)?back\b", "callback", text)
    for t in _TOKEN.findall(text):
        t = HINGLISH.get(t, t)
        t = SYNONYMS.get(t, t)
        if t not in STOP and len(t) > 1:
            toks.append(_stem(t))
    return toks


def _stem(t: str) -> str:
    # Tiny suffix stripper; enough for policy vocabulary, cheap at runtime.
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def char_ngrams(text: str, n_min: int = 3, n_max: int = 4) -> list[str]:
    s = " " + " ".join(normalise(text)) + " "
    return [s[i:i + n] for n in range(n_min, n_max + 1) for i in range(len(s) - n + 1)]
