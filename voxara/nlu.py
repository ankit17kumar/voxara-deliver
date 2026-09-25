"""Intent and slot extraction for delivery calls (English, Hindi, Hinglish).

Rules first, because delivery calls use a small, repetitive vocabulary and
rules are instant, free and auditable. `LLMFallback` in llm.py is consulted
only when no rule fires, which keeps LLM spend to the long tail.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
HINGLISH_MARKERS = {"hai", "hain", "nahi", "nahin", "kal", "haan", "ji", "mera", "mujhe",
                    "kya", "karo", "kar", "do", "dena", "chahiye", "bhai", "aap", "abhi",
                    "wala", "ghar", "theek", "thik", "parso", "aaj", "tha", "ko", "se", "ka", "ki"}

# Order matters: the first matching intent wins, most specific first.
INTENT_PATTERNS: list[tuple[str, str]] = [
    ("do_not_call", r"\b(do not call|don'?t call|call mat|phone mat|stop calling|dobara call mat)\b"),
    ("human", r"\b(human|real person|manager|supervisor|agent se|insaan|kisi (se|aadmi)|customer care|senior)\b"),
    ("fake_attempt", r"\b(never came|didn'?t come|did not come|nahi aaya|nahi aya|koi nahi aaya|"
                     r"didn'?t call|did not call|call nahi kiya|call hi nahi|fake|jhooth|jhoot|galat mark)\b"),
    ("refund", r"\b(refund|paise wapas|money back)\b"),
    ("recording", r"\b(record(ing)?|recorded)\b"),
    ("cancel", r"\b(cancel|radd|nahi chahiye|don'?t want|do not want|not needed|mat bhejo|"
               r"did not order|didn'?t order|order nahi kiya)\b"),
    ("refuse", r"\b(refused|reject(ed)?|mana kar|return kar|wapas bhej|nahi liya)\b"),
    ("prepaid", r"\b(prepaid|upi|online pay|pay online|payment link|gpay|phonepe|paytm)\b"),
    ("amount_query", r"\b(amount|kitne paise|kitna paisa|price|charge|zyada paise|too much|expensive|mehenga)\b"),
    ("address_change", r"\b(address|pata|pincode|pin code|landmark|flat|house no|ghar ka|"
                       r"near|ke paas|paas mein|alternate number|dusra number|doosra number)\b"),
    ("reschedule", r"\b(reschedule|another day|next day|dobara|phir se|fir se|re-?attempt|"
                   r"deliver (it )?(on|tomorrow|later)|bhej do|bhejo|le aao|aa jao|later|baad mein)\b"),
    ("delivery_time", r"\b(kab aayega|kab aaega|when will|what time|kitne baje|kab tak|status)\b"),
    ("deny", r"^\s*(no|nope|nahi|nahin|nai|na|galat|wrong)\b"),
    ("affirm", r"^\s*(yes|yeah|yep|haan|haa|ha|ji|hanji|haanji|ok|okay|theek|thik|sure|correct|sahi|bilkul|confirm)\b"),
    ("thanks", r"\b(thank|thanks|shukriya|dhanyavaad|dhanyawad)\b"),
    ("greet", r"^\s*(hello|hi|namaste|haan ji bolo|bolo)\b"),
]

DEVANAGARI_INTENTS = [
    ("human", r"(इंसान|मैनेजर|किसी से बात)"),
    ("cancel", r"(कैंसल|नहीं चाहिए|रद्द)"),
    ("reschedule", r"(कल|परसों|दोबारा)"),
    ("affirm", r"^\s*(हाँ|हां|जी|ठीक)"),
    ("deny", r"^\s*(नहीं|ना)"),
    ("thanks", r"(धन्यवाद|शुक्रिया)"),
]

WEEKDAYS = {"monday": 0, "somvar": 0, "tuesday": 1, "mangalvar": 1, "wednesday": 2, "budhvar": 2,
            "thursday": 3, "guruvar": 3, "friday": 4, "shukravar": 4, "saturday": 5, "shanivar": 5,
            "sunday": 6, "ravivar": 6, "itvaar": 6, "itwar": 6}
SLOTS = {"morning": "morning", "subah": "morning", "afternoon": "afternoon", "dopahar": "afternoon",
         "dopehar": "afternoon", "evening": "evening", "shaam": "evening", "sham": "evening",
         "सुबह": "morning", "दोपहर": "afternoon", "शाम": "evening"}


@dataclass
class NLUResult:
    intent: str
    lang: str
    slots: dict = field(default_factory=dict)
    confidence: float = 0.9
    source: str = "rules"


def detect_lang(text: str) -> str:
    if DEVANAGARI.search(text):
        return "hi"
    toks = re.findall(r"[a-z]+", text.lower())
    if toks and sum(t in HINGLISH_MARKERS for t in toks) / len(toks) >= 0.2:
        return "hi-en"
    return "en"


def extract_slots(text: str, today: date | None = None) -> dict:
    today = today or date.today()
    low = text.lower()
    slots: dict = {}

    if re.search(r"\b(day after tomorrow|parso|parson)\b|परसों", low):
        slots["date"] = today + timedelta(days=2)
    elif re.search(r"\b(tomorrow|kal|kl)\b|कल", low):
        slots["date"] = today + timedelta(days=1)
    elif re.search(r"\b(today|aaj|aj)\b|आज", low):
        slots["date"] = today
    else:
        for name, wd in WEEKDAYS.items():
            if re.search(rf"\b{name}\b", low):
                ahead = (wd - today.weekday()) % 7 or 7
                slots["date"] = today + timedelta(days=ahead)
                break
        m = re.search(r"\b(\d{1,2})\s*(tarikh|tareekh|th|st|nd|rd)\b", low)
        if "date" not in slots and m:
            d = int(m.group(1))
            try:
                cand = today.replace(day=d)
                if cand < today:
                    nxt = (today.replace(day=1) + timedelta(days=32)).replace(day=d)
                    cand = nxt
                slots["date"] = cand
            except ValueError:
                pass

    for word, slot in SLOTS.items():
        if re.search(rf"(?<![a-z]){word}(?![a-z])", low):
            slots["slot"] = slot
            break

    if m := re.search(r"(?<!\d)([1-9]\d{5})(?!\d)", text):
        slots["pincode"] = m.group(1)
    if m := re.search(r"(?<!\d)([6-9]\d{9})(?!\d)", re.sub(r"[\s-]", "", text)):
        slots["alt_phone"] = m.group(1)
    if m := re.search(r"\b(?:near|opposite|behind|next to)\s+([a-z0-9 ]{3,40}?)(?:[.,]|$)", low):
        slots["landmark"] = m.group(1).strip()
    elif m := re.search(r"([a-z0-9 ]{3,40}?)\s+(?:ke paas|ke saamne|ke peeche)\b", low):
        slots["landmark"] = m.group(1).strip()
    if m := re.search(r"\b(?:flat|house|h\.? ?no\.?|makan)\s*(?:no\.?|number)?\s*([a-z0-9/-]+)", low):
        slots["house_no"] = m.group(1)
    return slots


def parse(text: str, today: date | None = None) -> NLUResult:
    lang = detect_lang(text)
    low = text.lower().strip()
    slots = extract_slots(text, today)
    intent = None
    for name, pat in INTENT_PATTERNS:
        if re.search(pat, low):
            intent = name
            break
    if intent is None:
        for name, pat in DEVANAGARI_INTENTS:
            if re.search(pat, text):
                intent = name
                break
    if intent is None and ("date" in slots or "slot" in slots):
        intent = "provide_date"
    if intent is None and slots.keys() & {"pincode", "landmark", "alt_phone", "house_no"}:
        intent = "address_change"
    if intent is None:
        return NLUResult("unknown", lang, slots, 0.3)
    # A date inside a "yes" still counts as scheduling info
    if intent in ("affirm",) and "date" in slots:
        intent = "provide_date"
    return NLUResult(intent, lang, slots, 0.9)
