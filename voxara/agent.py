"""Voxara Deliver conversation orchestrator.

Per caller turn:  NLU -> emotion -> guardrails -> (RAG if needed) ->
workflow step -> OMS write -> response + speaking style.

Every call ends in exactly one outcome (deck principle: "a completed
action, a tracked exception, or a human handoff with context").
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from . import llm
from .compliance import Suppression, is_promotional, record_attempt, redact
from .emotion import EmotionDetector, EmotionState, ResponseStyle, style_for
from .nlu import NLUResult, parse
from .oms import IST, MockOMS, Order
from .rag import RagPipeline

WORKFLOW_FOR_STATUS = {"CONFIRMATION_PENDING": "cod_confirmation", "NDR": "ndr_recovery"}

REASON_HI = {
    "Customer not available": "aap us waqt available nahi the",
    "Customer refused": "order lene se mana kiya gaya",
    "Address incomplete": "address poora nahi tha",
    "Phone unreachable": "aapka phone nahi lag raha tha",
    "Customer asked to reschedule": "aapne dusre din ke liye kaha tha",
    "COD amount not ready": "cash ready nahi tha",
}
SLOT_TEXT = {"en": {"morning": "morning, 9 to 1", "afternoon": "afternoon, 1 to 5", "evening": "evening, 5 to 9"},
             "hi": {"morning": "subah 9 se 1", "afternoon": "dopahar 1 se 5", "evening": "shaam 5 se 9"}}

T = {
    "open_cod": {
        "en": "Hello {name}, this is the automated assistant from {brand}, calling about your order. "
              "This call may be recorded for quality. You ordered {product} for ₹{amount}, cash on delivery, "
              "to pincode {pincode}. Shall I confirm this order?",
        "hi": "Namaste {name} ji, main {brand} ki taraf se automated assistant bol rahi hoon, aapke order ke "
              "baare mein. Yeh call quality ke liye record ho sakti hai. Aapne {product} order kiya hai, "
              "₹{amount} cash on delivery, pincode {pincode} par. Kya main yeh order confirm kar doon?"},
    "open_ndr": {
        "en": "Hello {name}, this is the automated assistant from {brand}. This call may be recorded for quality. "
              "We could not deliver your {product} because {reason}. When should we try again, tomorrow or the day after?",
        "hi": "Namaste {name} ji, main {brand} ki taraf se automated assistant bol rahi hoon. Yeh call record ho "
              "sakti hai. Aapka {product} deliver nahi ho paaya kyunki {reason}. Hum dobara kab bhejein, kal ya parso?"},
    "open_ndr_address": {
        "en": "Hello {name}, this is the automated assistant from {brand}. This call may be recorded for quality. "
              "Our courier could not find your address for the {product}. Could you tell me a nearby landmark?",
        "hi": "Namaste {name} ji, main {brand} ki taraf se automated assistant bol rahi hoon. Yeh call record ho "
              "sakti hai. Courier ko aapka {product} deliver karte waqt address nahi mila. Kya aap paas ka koi "
              "landmark bata sakte hain?"},
    "ask_confirm": {"en": "Shall I confirm your order of {product}?", "hi": "Kya main {product} ka order confirm kar doon?"},
    "confirmed_offer_prepaid": {
        "en": "Your order is confirmed. Would you like a UPI link to pay online now instead of cash?",
        "hi": "Aapka order confirm ho gaya. Kya aap cash ki jagah abhi UPI se online pay karna chahenge? Main link bhej sakti hoon."},
    "prepaid_sent": {"en": "I have sent the UPI payment link on WhatsApp. Thank you, have a nice day!",
                     "hi": "Maine WhatsApp par UPI link bhej diya hai. Dhanyavaad, aapka din shubh ho!"},
    "cod_close": {"en": "No problem, you can pay cash on delivery. Thank you!",
                  "hi": "Koi baat nahi, aap delivery par cash de dijiyega. Dhanyavaad!"},
    "ask_cancel": {"en": "Would you like me to cancel this order?", "hi": "Kya main yeh order cancel kar doon?"},
    "cancelled": {"en": "Your order has been cancelled. There is no charge. Thank you for letting us know.",
                  "hi": "Aapka order cancel ho gaya hai, koi charge nahi lagega. Batane ke liye dhanyavaad."},
    "ask_date": {"en": "Which day works for you, tomorrow or the day after?", "hi": "Aapke liye kaunsa din theek rahega, kal ya parso?"},
    "ask_slot": {"en": "{day} works. Morning, afternoon or evening?", "hi": "{day} theek hai. Subah, dopahar ya shaam?"},
    "scheduled": {"en": "Done. We will try to deliver on {day}, {slot}. You will get an SMS before the courier arrives. Thank you!",
                  "hi": "Ho gaya. Hum {day} ko {slot} ke beech delivery ki koshish karenge. Courier aane se pehle SMS aayega. Dhanyavaad!"},
    "date_too_far": {"en": "I can schedule up to {max_day} at the latest. Shall I do {max_day}?",
                     "hi": "Main zyada se zyada {max_day} tak schedule kar sakti hoon. Kya {max_day} kar doon?"},
    "no_sunday": {"en": "Sunday delivery is not available in your area. Shall I do Monday?",
                  "hi": "Aapke area mein Sunday delivery nahi hoti. Kya Monday kar doon?"},
    "ask_refusal_reason": {"en": "May I ask why you did not accept the order?", "hi": "Kya main pooch sakti hoon ki aapne order kyun nahi liya?"},
    "refusal_offer": {"en": "Thank you for telling me. Would you like us to try delivering once more?",
                      "hi": "Batane ke liye shukriya. Kya hum ek baar aur delivery try karein?"},
    "rto": {"en": "Understood. We will return the order and you will not be charged. Thank you for your time.",
            "hi": "Samajh gayi. Hum order wapas bhej denge, aapse koi charge nahi liya jayega. Aapke samay ke liye dhanyavaad."},
    "sorry": {"en": "I am sorry about that.", "hi": "Iske liye maafi chahti hoon."},
    "fake_attempt": {"en": "I have registered a complaint against the courier and marked a priority delivery for tomorrow. Morning, afternoon or evening?",
                     "hi": "Maine courier ke khilaaf complaint darj kar di hai aur kal priority delivery laga di hai. Subah, dopahar ya shaam?"},
    "ask_landmark": {"en": "Could you tell me a nearby landmark, like a temple, school or shop?",
                     "hi": "Kya aap paas ka koi landmark bata sakte hain, jaise mandir, school ya dukaan?"},
    "address_updated": {"en": "Thank you, I have added {what} to your address.",
                        "hi": "Shukriya, maine address mein {what} jod diya hai."},
    "escalate": {"en": "I am connecting you to our team now. They will have all the details, so you will not need to repeat anything.",
                 "hi": "Main aapko abhi hamari team se connect kar rahi hoon. Unke paas saari details hongi, aapko kuch dobara batana nahi padega."},
    "refund_handoff": {"en": "This call is only for delivery, so I will pass your refund question to our support team. They will call you back within 2 working hours.",
                       "hi": "Yeh call sirf delivery ke liye hai, isliye main aapka refund sawaal support team ko bhej rahi hoon. Woh 2 working ghante mein call karenge."},
    "dnc": {"en": "Understood. We will not call you again about this order. Thank you.",
            "hi": "Theek hai, is order ke liye hum aapko dobara call nahi karenge. Dhanyavaad."},
    "recording_off": {"en": "Of course, I have stopped the recording.", "hi": "Zaroor, maine recording band kar di hai."},
    "didnt_get": {"en": "Sorry, I did not catch that.", "hi": "Maaf kijiye, main samajh nahi paayi."},
    "offer_human": {"en": "If you prefer, I can connect you to a person from our team.",
                    "hi": "Agar aap chahein toh main aapko hamari team se connect kar sakti hoon."},
    "amount_query": {"en": "The ₹{amount} includes the product price, taxes and any COD fee shown at checkout.",
                     "hi": "₹{amount} mein product ki keemat, tax aur checkout par dikhayi gayi COD fee shaamil hai."},
    "delivery_time": {"en": "We pass your preferred day and slot to the courier, but we cannot promise an exact time.",
                      "hi": "Hum aapka pasand ka din aur slot courier ko bhej dete hain, lekin exact time ki guarantee nahi de sakte."},
    "too_many_attempts": {"en": "This order has already had three attempts, so I will ask our team to call you and arrange it.",
                          "hi": "Is order ki teen koshish ho chuki hain, isliye hamari team aapko call karke arrange karegi."},
}


@dataclass
class Turn:
    role: str
    text: str
    ts: float
    meta: dict = field(default_factory=dict)


@dataclass
class CallSession:
    call_id: str
    order: Order
    workflow: str
    lang: str
    state: str
    turns: list[Turn] = field(default_factory=list)
    emotions: list[EmotionState] = field(default_factory=list)
    outcome: str | None = None
    ended: bool = False
    low_conf: int = 0
    prepaid_offered: bool = False
    human_offered: bool = False
    recording: bool = True
    apology_needed: bool = False
    pending_date: date | None = None
    today: date = field(default_factory=lambda: datetime.now(IST).date())
    latencies: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        return {"call_id": self.call_id, "order_id": self.order.order_id, "workflow": self.workflow,
                "lang": self.lang, "state": self.state, "outcome": self.outcome, "ended": self.ended,
                "turns": [{"role": t.role, "text": t.text, **t.meta} for t in self.turns],
                "emotion_trend": [e.label for e in self.emotions],
                "avg_latency_ms": round(sum(self.latencies) / len(self.latencies), 2) if self.latencies else None}


@dataclass
class AgentReply:
    text: str
    style: dict
    emotion: dict | None
    nlu: dict | None
    rag: dict | None
    actions: list[dict]
    outcome: str | None
    ended: bool
    latency_ms: float


class DeliverAgent:
    def __init__(self, oms: MockOMS, rag: RagPipeline, detector: EmotionDetector | None = None,
                 suppression: Suppression | None = None):
        self.oms, self.rag = oms, rag
        self.detector = detector or EmotionDetector()
        self.sup = suppression or Suppression()
        self.calls: dict[str, CallSession] = {}

    # ---------- helpers ----------
    @staticmethod
    def _l(lang: str) -> str:
        return "hi" if lang.startswith("hi") else "en"

    def _t(self, s: CallSession, key: str, **kw) -> str:
        return T[key][self._l(s.lang)].format(**kw)

    def _day(self, s: CallSession, d: date) -> str:
        if d == s.today + timedelta(days=1):
            prefix = "kal" if self._l(s.lang) == "hi" else "tomorrow"
            return f"{prefix}, {d.strftime('%A')} {d.day} {d.strftime('%B')}"
        return f"{d.strftime('%A')} {d.day} {d.strftime('%B')}"

    def _finish(self, s: CallSession, outcome: str, action: str | None = None, **fields) -> list[dict]:
        s.outcome, s.ended = outcome, True
        acts = []
        if action:
            self.oms.apply_outcome(s.order.order_id, action, **fields)
            acts.append({"type": "oms_update", "action": action, "fields": {k: str(v) for k, v in fields.items()}})
        return acts

    # ---------- call lifecycle ----------
    def start_call(self, order_id: str, lang: str | None = None, today: date | None = None) -> tuple[CallSession, AgentReply]:
        order = self.oms.get_order(order_id)
        if order is None:
            raise KeyError(order_id)
        workflow = WORKFLOW_FOR_STATUS.get(order.status)
        if workflow is None:
            raise ValueError(f"No Deliver workflow for status {order.status}")
        s = CallSession(uuid.uuid4().hex[:10], order, workflow, lang or order.lang_pref, "")
        if today:
            s.today = today
        record_attempt(order_id, self.sup)
        first = order.customer_name.split()[0]
        common = dict(name=first, brand=order.brand, product=order.product,
                      amount=int(order.amount), pincode=order.pincode)
        if workflow == "cod_confirmation":
            s.state, text = "confirm", self._t(s, "open_cod", **common)
        elif order.ndr_reason == "Address incomplete":
            s.state, text = "ask_landmark", self._t(s, "open_ndr_address", **common)
        else:
            reason = REASON_HI.get(order.ndr_reason, order.ndr_reason) if self._l(s.lang) == "hi" \
                else order.ndr_reason.lower()
            s.state, text = "ask_date", self._t(s, "open_ndr", reason=reason, **common)
        assert not is_promotional(text)
        s.turns.append(Turn("agent", text, time.time(), {"state": s.state}))
        self.calls[s.call_id] = s
        style = style_for(EmotionState(), [], s.lang).to_dict()
        return s, AgentReply(text, style, None, None, None, [], None, False, 0.0)

    def handle(self, call_id: str, text: str, prosody: dict | None = None) -> AgentReply:
        t0 = time.perf_counter()
        s = self.calls[call_id]
        if s.ended:
            return AgentReply("", {}, None, None, None, [], s.outcome, True, 0.0)

        nlu = parse(text, s.today)
        if nlu.intent == "unknown" and (llm_intent := llm.classify_intent(text)):
            nlu.intent, nlu.source = llm_intent, "llm"
        if nlu.lang != "en" and s.lang == "en":
            s.lang = nlu.lang          # follow the caller if they switch to Hindi
        emo = self.detector.detect(text, prosody, s.emotions)
        style = style_for(emo, s.emotions, s.lang, len(s.turns))
        prev = s.emotions[-1] if s.emotions else None
        # Acknowledge a feeling once, not every turn: repeated empathy sounds scripted.
        acknowledge = bool(style.acknowledgement) and (
            prev is None or prev.label != emo.label or emo.trend == "escalating")
        s.emotions.append(emo)
        s.turns.append(Turn("customer", redact(text) if s.recording else "[not recorded]", time.time(),
                            {"intent": nlu.intent, "emotion": emo.label, "intensity": emo.intensity}))

        rag_info = None
        body, actions = self._route(s, nlu, text, style)
        if isinstance(body, tuple):
            body, rag_info = body

        parts = []
        if acknowledge:
            parts.append(style.acknowledgement)
        elif s.apology_needed:
            parts.append(self._t(s, "sorry"))
        s.apology_needed = False
        parts.append(body)
        if style.offer_human and not s.ended and not s.human_offered:
            parts.append(self._t(s, "offer_human"))
            s.human_offered = True
        reply = " ".join(p for p in parts if p).strip()
        assert not is_promotional(reply), "transactional calls must not carry promotions"

        latency = (time.perf_counter() - t0) * 1000
        s.latencies.append(latency)
        s.turns.append(Turn("agent", reply, time.time(), {"state": s.state, "outcome": s.outcome}))
        return AgentReply(reply, style.to_dict(), emo.to_dict(),
                          {"intent": nlu.intent, "lang": nlu.lang, "source": nlu.source,
                           "slots": {k: str(v) for k, v in nlu.slots.items()}},
                          rag_info, actions, s.outcome, s.ended, round(latency, 2))

    # ---------- routing ----------
    def _route(self, s: CallSession, nlu: NLUResult, text: str, style: ResponseStyle):
        i = nlu.intent
        # Global handlers first: these apply in any state.
        very_angry_stuck = style.reason == "very angry" and i in ("unknown", "deny")
        if style.escalate_now or i == "human" or very_angry_stuck:
            reason = "customer asked for human" if i == "human" else style.reason
            return self._t(s, "escalate"), self._escalate(s, reason)
        if i == "do_not_call":
            self.sup.suppress(s.order.phone)
            return self._t(s, "dnc"), self._finish(s, "DO_NOT_CALL", "suppress_number")
        if i == "refund":
            return self._t(s, "refund_handoff"), self._escalate(s, "refund question", outcome="CALLBACK_REQUESTED")
        if i == "recording":
            s.recording = False
            return self._t(s, "recording_off") + " " + self._reprompt(s), []
        if s.state == "refusal_reason":        # any free-text answer is the reason
            return self._state_refusal_reason(s, nlu)
        if i in ("amount_query", "delivery_time") or (i == "unknown"):
            return self._policy_answer(s, nlu, text, style)

        handler = getattr(self, f"_state_{s.state}", None)
        return handler(s, nlu) if handler else self._policy_answer(s, nlu, text, style)

    def _escalate(self, s: CallSession, reason: str, outcome: str = "ESCALATED") -> list[dict]:
        ctx = {"reason": reason, "emotion_trend": ",".join(e.label for e in s.emotions),
               "last_request": s.turns[-1].text if s.turns else "", "state": s.state}
        acts = self._finish(s, outcome, "create_handoff_task", notes=f"handoff: {reason}")
        acts.append({"type": "handoff", "context": ctx})
        return acts

    def _policy_answer(self, s: CallSession, nlu: NLUResult, text: str, style: ResponseStyle):
        r = self.rag.retrieve(text, nlu.intent, s.workflow, None)
        info = {"citations": r.citations, "confidence": r.confidence, "latency_ms": round(r.latency_ms, 2),
                "cached": r.cached, "grounded": r.grounded}
        if r.grounded:
            s.low_conf = 0
            if nlu.intent in ("amount_query", "delivery_time"):
                # Frequent, high-stakes answers are pre-approved templates;
                # retrieval still runs so the answer is cited in the audit log.
                ans = self._t(s, nlu.intent, amount=int(s.order.amount))
            else:
                ans = llm.phrase_policy_answer(text, r.context, s.lang, style.tone)
            return (f"{ans} {self._reprompt(s)}", info), []
        s.low_conf += 1
        if s.low_conf >= 2:
            return (self._t(s, "escalate"), info), self._escalate(s, "low confidence twice")
        return (f"{self._t(s, 'didnt_get')} {self._reprompt(s)}", info), []

    def _reprompt(self, s: CallSession) -> str:
        return {
            "confirm": self._t(s, "ask_confirm", product=s.order.product),
            "prepaid_offer": "",
            "confirm_cancel": self._t(s, "ask_cancel"),
            "ask_date": self._t(s, "ask_date"),
            "ask_slot": self._t(s, "ask_slot", day=self._day(s, s.pending_date)) if s.pending_date else "",
            "refusal_reason": self._t(s, "ask_refusal_reason"),
            "refusal_offer": self._t(s, "refusal_offer"),
            "ask_landmark": self._t(s, "ask_landmark"),
        }.get(s.state, "")

    # ---------- COD confirmation ----------
    def _state_confirm(self, s: CallSession, nlu: NLUResult):
        if nlu.intent in ("affirm", "thanks"):
            self.oms.apply_outcome(s.order.order_id, "confirm_order", status="CONFIRMED")
            s.state, s.prepaid_offered = "prepaid_offer", True
            return self._t(s, "confirmed_offer_prepaid"), [{"type": "oms_update", "action": "confirm_order"}]
        if nlu.intent == "prepaid":
            return self._t(s, "prepaid_sent"), self._finish(
                s, "CONFIRMED_PREPAID", "confirm_order_prepaid", status="CONFIRMED", payment="PREPAID") + [
                {"type": "whatsapp", "template": "upi_payment_link"}]
        if nlu.intent in ("deny", "cancel", "refuse"):
            s.state = "confirm_cancel"
            return self._t(s, "ask_cancel"), []
        if nlu.intent == "address_change":
            return self._address(s, nlu, then="confirm")
        return f"{self._t(s, 'didnt_get')} {self._reprompt(s)}", []

    def _state_prepaid_offer(self, s: CallSession, nlu: NLUResult):
        if nlu.intent in ("affirm", "prepaid"):
            return self._t(s, "prepaid_sent"), self._finish(
                s, "CONFIRMED_PREPAID", "switch_to_prepaid", payment="PREPAID") + [
                {"type": "whatsapp", "template": "upi_payment_link"}]
        # Offer is made once only (cod_policy.md); anything else closes as COD.
        return self._t(s, "cod_close"), self._finish(s, "CONFIRMED_COD")

    def _state_confirm_cancel(self, s: CallSession, nlu: NLUResult):
        if nlu.intent in ("affirm", "cancel"):
            return self._t(s, "cancelled"), self._finish(s, "CANCELLED", "cancel_order", status="CANCELLED")
        s.state = "confirm"
        return self._t(s, "ask_confirm", product=s.order.product), []

    # ---------- NDR recovery ----------
    def _state_ask_date(self, s: CallSession, nlu: NLUResult):
        if nlu.intent == "fake_attempt":
            s.pending_date = s.today + timedelta(days=1)
            s.state, s.apology_needed = "ask_slot", True
            return self._t(s, "fake_attempt"), [{"type": "courier_complaint", "kind": "fake_attempt"}]
        if nlu.intent in ("refuse", "cancel", "deny") and "date" not in nlu.slots:
            s.state = "refusal_reason"
            return self._t(s, "ask_refusal_reason"), []
        if nlu.intent == "address_change":
            return self._address(s, nlu, then="ask_date")
        if "date" in nlu.slots:
            return self._take_date(s, nlu)
        return self._t(s, "ask_date"), []

    def _take_date(self, s: CallSession, nlu: NLUResult):
        d: date = nlu.slots["date"]
        if s.order.attempts >= 3:
            return self._t(s, "too_many_attempts"), self._escalate(s, "max attempts reached")
        max_day = s.today + timedelta(days=3)
        if d > max_day:
            s.pending_date, s.state = max_day, "confirm_far_date"
            return self._t(s, "date_too_far", max_day=self._day(s, max_day)), []
        if d <= s.today:
            d = s.today + timedelta(days=1)
        if d.weekday() == 6 and not s.order.is_metro:
            s.pending_date, s.state = d + timedelta(days=1), "confirm_far_date"
            return self._t(s, "no_sunday"), []
        s.pending_date = d
        if "slot" in nlu.slots:
            return self._schedule(s, nlu.slots["slot"])
        s.state = "ask_slot"
        return self._t(s, "ask_slot", day=self._day(s, d)), []

    def _state_confirm_far_date(self, s: CallSession, nlu: NLUResult):
        if nlu.intent == "affirm":
            s.state = "ask_slot"
            return self._t(s, "ask_slot", day=self._day(s, s.pending_date)), []
        if "date" in nlu.slots:
            return self._take_date(s, nlu)
        s.state = "ask_date"
        return self._t(s, "ask_date"), []

    def _state_ask_slot(self, s: CallSession, nlu: NLUResult):
        if "slot" in nlu.slots:
            return self._schedule(s, nlu.slots["slot"])
        if "date" in nlu.slots:
            return self._take_date(s, nlu)
        if nlu.intent == "affirm":      # "any time is fine"
            return self._schedule(s, "afternoon")
        return self._t(s, "ask_slot", day=self._day(s, s.pending_date)), []

    def _schedule(self, s: CallSession, slot: str):
        d = s.pending_date
        acts = self._finish(s, "REATTEMPT_SCHEDULED", "schedule_reattempt", status="REATTEMPT_SCHEDULED",
                            reattempt_date=d.isoformat(), reattempt_slot=slot, attempts=s.order.attempts)
        acts.append({"type": "sms", "template": "reattempt_confirmation"})
        return self._t(s, "scheduled", day=self._day(s, d), slot=SLOT_TEXT[self._l(s.lang)][slot]), acts

    def _state_refusal_reason(self, s: CallSession, nlu: NLUResult):
        reason = s.turns[-1].text
        self.oms.apply_outcome(s.order.order_id, "record_refusal_reason", notes=f"refusal: {reason}")
        if nlu.intent in ("reschedule", "provide_date", "affirm") or "date" in nlu.slots:
            s.state = "ask_date"
            return self._state_ask_date(s, nlu) if "date" in nlu.slots else (self._t(s, "ask_date"), [])
        s.state = "refusal_offer"
        return self._t(s, "refusal_offer"), [{"type": "oms_update", "action": "record_refusal_reason"}]

    def _state_refusal_offer(self, s: CallSession, nlu: NLUResult):
        if nlu.intent in ("affirm", "reschedule", "provide_date"):
            s.state = "ask_date"
            return self._take_date(s, nlu) if "date" in nlu.slots else (self._t(s, "ask_date"), [])
        # Offer made once; never repeat or pressure (ndr_policy.md).
        return self._t(s, "rto"), self._finish(s, "RTO_CONFIRMED", "mark_rto", status="RTO")

    def _state_ask_landmark(self, s: CallSession, nlu: NLUResult):
        return self._address(s, nlu, then="ask_date")

    def _address(self, s: CallSession, nlu: NLUResult, then: str):
        fields, said = {}, []
        o = s.order
        if lm := nlu.slots.get("landmark"):
            fields["landmark"] = lm.title()
            said.append(f"landmark {lm.title()}")
        if hn := nlu.slots.get("house_no"):
            fields["address_line"] = f"{hn}, {o.address_line}"
            said.append(f"house number {hn}")
        if pin := nlu.slots.get("pincode"):
            if pin[:3] == o.pincode[:3]:
                fields["pincode"] = pin
                said.append(f"pincode {pin}")
            else:
                return self._t(s, "escalate"), self._escalate(s, "pincode change to a different city")
        if ph := nlu.slots.get("alt_phone"):
            fields["notes"] = f"alt phone XXXXXX{ph[-4:]}"
            said.append(f"alternate number ending {ph[-4:]}")
        if not fields:
            if s.state != "ask_landmark" and nlu.intent == "address_change":
                s.state = "ask_landmark"
            return self._t(s, "ask_landmark"), []
        self.oms.apply_outcome(o.order_id, "update_address", **fields)
        for k, v in fields.items():
            setattr(o, k, v)
        s.state = then
        joiner = " aur " if self._l(s.lang) == "hi" else " and "
        text = self._t(s, "address_updated", what=joiner.join(said)) + " " + self._reprompt(s)
        return text.strip(), [{"type": "oms_update", "action": "update_address", "fields": fields}]
