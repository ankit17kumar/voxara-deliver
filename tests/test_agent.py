from datetime import date
from pathlib import Path

import pytest

from voxara.agent import DeliverAgent
from voxara.emotion import EmotionDetector
from voxara.nlu import parse
from voxara.oms import MockOMS, Order
from voxara.rag import RagPipeline

KB = Path(__file__).resolve().parents[1] / "voxara" / "kb"
TODAY = date(2026, 9, 25)          # a Friday


def order(oid, status, reason="", city="Jaipur", pin="302017", lang="hi-en", attempts=1):
    return Order(oid, "UrbanKurta", "Priya Sharma", "9876543210", lang, city, pin, "12, Malviya Nagar",
                 "", "Cotton kurta set", 1299.0, "COD", status, reason, attempts)


@pytest.fixture
def agent():
    oms = MockOMS()
    for o in [order("C1", "CONFIRMATION_PENDING"), order("N1", "NDR", "Customer not available"),
              order("N2", "NDR", "Address incomplete"), order("N3", "NDR", "Customer refused"),
              order("N4", "NDR", "Customer not available", attempts=3),
              order("M1", "NDR", "Customer not available", city="Mumbai", pin="400053")]:
        oms.upsert(o)
    return DeliverAgent(oms, RagPipeline(KB))


def talk(agent, oid, *lines):
    s, _ = agent.start_call(oid, today=TODAY)
    r = None
    for line in lines:
        r = agent.handle(s.call_id, line)
        if r.ended:
            break
    return s, r


def test_cod_confirm_then_decline_prepaid(agent):
    s, r = talk(agent, "C1", "haan ji confirm karo", "nahi cash hi dunga")
    assert r.outcome == "CONFIRMED_COD"
    assert agent.oms.get_order("C1").status == "CONFIRMED"


def test_cod_switch_to_prepaid(agent):
    s, r = talk(agent, "C1", "yes", "haan UPI link bhej do")
    assert r.outcome == "CONFIRMED_PREPAID"
    assert agent.oms.get_order("C1").payment == "PREPAID"


def test_cod_cancel(agent):
    s, r = talk(agent, "C1", "mujhe nahi chahiye", "haan")
    assert r.outcome == "CANCELLED"


def test_amount_question_is_answered_and_cited(agent):
    s, _ = agent.start_call("C1", today=TODAY)
    r = agent.handle(s.call_id, "itne zyada paise kyun?")
    assert "1299" in r.text and r.rag["citations"][0] == "cod_policy#amount-questions"


def test_ndr_hinglish_one_shot_schedule(agent):
    s, r = talk(agent, "N1", "kal shaam ko bhej do")
    o = agent.oms.get_order("N1")
    assert r.outcome == "REATTEMPT_SCHEDULED"
    assert o.reattempt_date == "2026-09-26" and o.reattempt_slot == "evening"


def test_ndr_date_beyond_three_days_is_capped(agent):
    s, _ = agent.start_call("N1", today=TODAY)
    r = agent.handle(s.call_id, "deliver it on Thursday")          # 6 days out
    assert s.state == "confirm_far_date"
    r = agent.handle(s.call_id, "ok")
    r = agent.handle(s.call_id, "morning")
    assert agent.oms.get_order("N1").reattempt_date == "2026-09-28"


def test_no_sunday_for_non_metro(agent):
    s, _ = agent.start_call("N1", today=TODAY)
    agent.handle(s.call_id, "sunday ko bhejo")                     # Jaipur is not metro
    assert s.state == "confirm_far_date" and s.pending_date.weekday() == 0


def test_sunday_ok_for_metro(agent):
    s, r = talk(agent, "M1", "sunday morning")
    assert r.outcome == "REATTEMPT_SCHEDULED"


def test_address_incomplete_landmark(agent):
    s, r = talk(agent, "N2", "Hanuman mandir ke paas", "kal subah")
    o = agent.oms.get_order("N2")
    assert "Hanuman Mandir" in o.landmark and r.outcome == "REATTEMPT_SCHEDULED"


def test_refusal_asks_reason_once_then_rto(agent):
    s, r = talk(agent, "N3", "I refused it", "the product came late", "no thanks")
    assert r.outcome == "RTO_CONFIRMED"
    assert agent.oms.get_order("N3").status == "RTO"
    asks = [t for t in s.turns if t.role == "agent" and "try" in t.text.lower() or "aur delivery" in t.text]
    assert len(asks) == 1                                           # offer made once, no pressure


def test_fake_attempt_complaint_and_priority(agent):
    s, r = talk(agent, "N1", "delivery boy ne call hi nahi kiya, bahut bakwas service hai!", "evening")
    assert r.outcome == "REATTEMPT_SCHEDULED"                       # angry but actionable: resolved, not dumped


def test_sustained_anger_escalates(agent):
    s, r = talk(agent, "N1", "this is USELESS!!", "I am SO ANGRY, worst service ever!")
    assert r.outcome == "ESCALATED"
    assert any(a["type"] == "handoff" for a in r.actions)


def test_legal_threat_escalates_immediately(agent):
    s, r = talk(agent, "N1", "I will go to consumer court, this is fraud")
    assert r.outcome == "ESCALATED"


def test_human_request(agent):
    s, r = talk(agent, "N1", "mujhe kisi insaan se baat karni hai")
    assert r.outcome == "ESCALATED"


def test_off_topic_twice_escalates_not_hallucinates(agent):
    s, _ = agent.start_call("N1", today=TODAY)
    r1 = agent.handle(s.call_id, "who is the PM")
    assert "9 PM" not in r1.text
    r2 = agent.handle(s.call_id, "what's the weather")
    assert r2.outcome == "ESCALATED"


def test_max_attempts_escalates(agent):
    s, r = talk(agent, "N4", "kal bhej do")
    assert r.outcome == "ESCALATED"


def test_do_not_call(agent):
    s, r = talk(agent, "N1", "please do not call me again")
    assert r.outcome == "DO_NOT_CALL"
    assert "9876543210" in agent.sup.numbers


def test_transcript_masks_phone(agent):
    s, _ = agent.start_call("N2", today=TODAY)
    agent.handle(s.call_id, "call my brother on 9812345678")
    assert "9812345678" not in " ".join(t.text for t in s.turns)


# ---- NLU and emotion units ----

@pytest.mark.parametrize("text,intent", [
    ("haan ji", "affirm"), ("nahi", "deny"), ("kal bhej do", "reschedule"),
    ("cancel kar do", "cancel"), ("online pay karna hai", "prepaid"),
    ("koi nahi aaya ghar pe", "fake_attempt"), ("no thanks", "deny"),
    ("मुझे नहीं चाहिए", "cancel"), ("manager se baat karao", "human"),
])
def test_intents(text, intent):
    assert parse(text, TODAY).intent == intent


@pytest.mark.parametrize("text,label", [
    ("bahut bakwas service hai", "angry"), ("third time calling, still waiting", "frustrated"),
    ("it's my daughter's birthday gift, urgent please", "anxious"),
    ("matlab? samajh nahi aaya", "confused"), ("thank you so much, great", "happy"),
    ("kal bhej do", "neutral"), ("not happy with this", "frustrated"),
])
def test_emotions(text, label):
    assert EmotionDetector().detect(text).label == label


def test_prosody_raises_anger():
    d = EmotionDetector()
    flat = d.detect("why is it late")
    loud = d.detect("why is it late", {"energy": 0.9, "pitch_var": 0.8})
    assert loud.scores["angry"] > flat.scores["angry"]
