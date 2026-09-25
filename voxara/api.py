"""HTTP API and dashboard server.

Run:  uvicorn voxara.api:app --reload      then open http://localhost:8000
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import llm
from .agent import DeliverAgent
from .compliance import can_call
from .emotion import EmotionDetector
from .oms import MockOMS, Order, seed
from .rag import RagPipeline

ROOT = Path(__file__).resolve().parent
WEB = ROOT.parent / "web"
# Rough value of one avoided return, used only for the dashboard estimate.
# Earlier research: ₹150-300 per RTO in India (vendor figures, unverified).
RTO_COST_ESTIMATE = int(os.environ.get("VOXARA_RTO_COST", "200"))

app = FastAPI(title="Voxara Deliver", version="0.1.0")
state: dict = {}


def _boot():
    oms = MockOMS(os.environ.get("VOXARA_DB", ":memory:"))
    if not oms.list_orders():
        seed(oms)
    state["agent"] = DeliverAgent(oms, RagPipeline(ROOT / "kb"), EmotionDetector())


_boot()


def agent() -> DeliverAgent:
    return state["agent"]


class StartCall(BaseModel):
    order_id: str
    lang: str | None = None


class TurnIn(BaseModel):
    text: str
    prosody: dict | None = None


class RagQuery(BaseModel):
    query: str
    intent: str | None = None
    workflow: str | None = None


class EmotionIn(BaseModel):
    text: str
    prosody: dict | None = None


class NdrWebhook(BaseModel):
    """Shape loosely modelled on courier-aggregator NDR webhooks."""
    order_id: str
    brand: str
    customer_name: str
    phone: str
    city: str
    pincode: str
    address_line: str
    product: str
    amount: float
    ndr_reason: str
    attempts: int = 1
    lang_pref: str = "hi-en"


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "llm": "anthropic" if llm.enabled() else "offline (extractive)",
            "kb_chunks": len(agent().rag.chunks)}


@app.get("/api/orders")
def orders():
    return [o.to_dict() for o in agent().oms.list_orders()]


@app.post("/api/calls")
def start_call(body: StartCall):
    try:
        s, r = agent().start_call(body.order_id, body.lang)
    except KeyError:
        raise HTTPException(404, "order not found")
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"call_id": s.call_id, "workflow": s.workflow, "lang": s.lang, "reply": r.__dict__}


@app.post("/api/calls/{call_id}/turn")
def turn(call_id: str, body: TurnIn):
    if call_id not in agent().calls:
        raise HTTPException(404, "call not found")
    return agent().handle(call_id, body.text, body.prosody).__dict__


@app.get("/api/calls/{call_id}")
def call(call_id: str):
    s = agent().calls.get(call_id)
    if not s:
        raise HTTPException(404, "call not found")
    return s.summary()


@app.get("/api/calls")
def calls():
    return [s.summary() for s in agent().calls.values()]


@app.post("/api/webhooks/ndr")
def ndr_webhook(body: NdrWebhook):
    """Ingest an NDR from an OMS or courier aggregator and decide whether to call now."""
    a = agent()
    a.oms.upsert(Order(body.order_id, body.brand, body.customer_name, body.phone, body.lang_pref,
                       body.city, body.pincode, body.address_line, "", body.product, body.amount,
                       "COD", "NDR", body.ndr_reason, body.attempts))
    ok, why = can_call(body.phone, body.order_id, a.sup)
    return {"accepted": True, "call_now": ok, "reason": why}


@app.post("/api/rag/search")
def rag_search(body: RagQuery):
    r = agent().rag.retrieve(body.query, body.intent, body.workflow)
    return {"context": r.context, "citations": r.citations, "confidence": r.confidence,
            "grounded": r.grounded, "latency_ms": round(r.latency_ms, 3), "cached": r.cached, "hits": r.hits}


@app.post("/api/emotion")
def emotion(body: EmotionIn):
    return agent().detector.detect(body.text, body.prosody).to_dict()


@app.get("/api/kpis")
def kpis():
    a = agent()
    done = [s for s in a.calls.values() if s.ended]
    outcomes = Counter(s.outcome for s in done)
    ndr_done = [s for s in done if s.workflow == "ndr_recovery"]
    recovered = sum(1 for s in ndr_done if s.outcome == "REATTEMPT_SCHEDULED")
    handoffs = outcomes["ESCALATED"] + outcomes["CALLBACK_REQUESTED"]
    lat = [x for s in a.calls.values() for x in s.latencies]
    emo = Counter(e.label for s in a.calls.values() for e in s.emotions)
    pct = lambda n, d: round(100 * n / d, 1) if d else None
    return {
        "calls_total": len(a.calls), "calls_completed": len(done),
        "containment_pct": pct(len(done) - handoffs, len(done)),
        "action_completion_pct": pct(sum(1 for s in done if s.outcome not in ("ESCALATED", None)), len(done)),
        "ndr_recovery_pct": pct(recovered, len(ndr_done)),
        "estimated_rto_value_saved_inr": recovered * RTO_COST_ESTIMATE,
        "avg_turn_latency_ms": round(sum(lat) / len(lat), 2) if lat else None,
        "outcomes": dict(outcomes), "emotions": dict(emo), "rag": a.rag.stats,
    }


@app.get("/api/events")
def events():
    return agent().oms.events()


@app.post("/api/reset")
def reset():
    _boot()
    return {"ok": True}
