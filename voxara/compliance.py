"""Calling-compliance guardrails for India (see kb/compliance_policy.md).

These are hard checks in code, not prompt instructions, so no model output
can bypass them. They cover what TRAI and the DPDP Act make risky for a
transactional delivery call; they are not legal advice.
"""
from __future__ import annotations

import re
from datetime import datetime, time, timedelta

from .oms import IST

CALL_WINDOW = (time(9, 0), time(21, 0))
MAX_ATTEMPTS_PER_NDR = 2
MIN_GAP = timedelta(hours=3)
PROMO_WORDS = re.compile(r"\b(offer|sale|discount on (your )?next|new collection|buy now|coupon)\b", re.I)


class Suppression:
    def __init__(self):
        self.numbers: set[str] = set()
        self.attempts: dict[str, list[datetime]] = {}

    def suppress(self, phone: str):
        self.numbers.add(phone)


def can_call(phone: str, order_id: str, sup: Suppression, now: datetime | None = None) -> tuple[bool, str]:
    now = now or datetime.now(IST)
    if phone in sup.numbers:
        return False, "number suppressed (customer asked not to be called)"
    if not (CALL_WINDOW[0] <= now.timetz().replace(tzinfo=None) < CALL_WINDOW[1]):
        return False, "outside 9 AM to 9 PM IST; queued for next morning"
    past = sup.attempts.get(order_id, [])
    if len(past) >= MAX_ATTEMPTS_PER_NDR:
        return False, "max 2 attempts for this NDR reached"
    if past and now - past[-1] < MIN_GAP:
        return False, "last attempt under 3 hours ago"
    return True, "ok"


def record_attempt(order_id: str, sup: Suppression, now: datetime | None = None):
    sup.attempts.setdefault(order_id, []).append(now or datetime.now(IST))


def redact(text: str) -> str:
    """Mask phone numbers and long digit runs before storing a transcript."""
    text = re.sub(r"(?<!\d)([6-9]\d{5})(\d{4})(?!\d)", r"XXXXXX\2", text)
    text = re.sub(r"(?<!\d)(\d{4})[ -]?(\d{4})[ -]?(\d{4})[ -]?(\d{4})(?!\d)", "XXXX-XXXX-XXXX-\\4", text)
    return text


def is_promotional(text: str) -> bool:
    return bool(PROMO_WORDS.search(text))
