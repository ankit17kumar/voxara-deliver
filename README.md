# Voxara Deliver (MVP)

An AI voice agent that handles COD confirmation and failed-delivery (NDR) recovery calls for Indian e-commerce. It speaks English, Hindi and Hinglish, notices how the caller feels and adapts, answers policy questions from a grounded knowledge base, and writes every outcome back to the order system.

## Run it

```bash
pip install -r requirements.txt
uvicorn voxara.api:app --reload
# open http://localhost:8000  (Chrome or Edge for the microphone)
```

Pick an order, press **Call**, then type or speak as the customer. The right-hand panel shows the detected emotion, how the agent changes its tone, what it understood, which policy it cited, and what it wrote to the OMS.

```bash
python -m pytest -q          # 35 behaviour tests
python -m tests.eval_rag     # retrieval accuracy and latency
```

Everything runs offline with no API keys. Set `ANTHROPIC_API_KEY` to let a small LLM rephrase policy answers and classify utterances the rules miss (see `voxara/llm.py`).

## What's in the box

| Path | What it does |
|---|---|
| `voxara/agent.py` | Conversation orchestrator and the COD / NDR workflow state machine |
| `voxara/emotion.py` | Emotion and sentiment detection (text + optional prosody) and the response-style policy |
| `voxara/rag/` | Voice-optimised hybrid RAG: chunker, BM25 + dense index, pipeline with routing, cache and confidence gate |
| `voxara/kb/` | Delivery policies the agent is allowed to speak from |
| `voxara/nlu.py` | Intent and slot extraction for English, Hindi and Hinglish (dates, slots, pincode, landmark, phone) |
| `voxara/compliance.py` | Calling window, attempt limits, do-not-call suppression, PII redaction, no-promotion check |
| `voxara/oms.py` | Mock OMS (SQLite) with the adapter interface a real OMS integration implements |
| `voxara/api.py` | FastAPI: calls, turns, KPIs, NDR webhook, RAG and emotion endpoints |
| `web/index.html` | Dashboard and browser voice demo |

## Workflows

**COD confirmation:** confirm product, amount and pincode, offer a UPI link once, and cancel cleanly if the customer says so.

**NDR recovery:** reattempt within 3 days (Sunday only in metros, max 3 attempts), a slot preference, fake-attempt complaints with priority redelivery, landmark and address fixes, asking the refusal reason once and never pressuring, then RTO.

Every call ends in one outcome: `CONFIRMED_COD`, `CONFIRMED_PREPAID`, `CANCELLED`, `REATTEMPT_SCHEDULED`, `RTO_CONFIRMED`, `ESCALATED`, `CALLBACK_REQUESTED` or `DO_NOT_CALL`.

## Emotion layer

For every turn it detects `angry`, `frustrated`, `anxious`, `confused`, `happy` or `neutral`, with an intensity, a sentiment score (-1 to 1) and the trend across the call. It uses cues from the words, including romanised Hindi and Devanagari, negation ("khush nahi"), intensifiers ("bahut") and caps or exclamation marks. When the browser or the telephony layer sends them, it also uses loudness, variability and speech rate.

Each emotion maps to a speaking style: what to acknowledge first, TTS speed and pitch, reply length, whether to offer a human, and whether to hand off now. It acknowledges a feeling once, not every turn. It hands off straight away on a legal or fraud threat, on anger in two turns running, or when the caller is very angry and the agent can't act on what they asked. An angry caller with a request the agent can act on (for example "the courier never came") gets empathy and a fix, not a transfer.

## RAG pipeline: why it's built this way

1. **Route first.** Order facts come from the OMS, never from retrieved text. RAG only handles policy.
2. **Intent-aware query rewrite** adds policy vocabulary to short, noisy transcripts.
3. **Workflow metadata filter** runs before scoring.
4. **Hybrid BM25 + hashed char-n-gram dense retrieval**, fused with RRF and followed by a cheap lexical rerank. The Hinglish normaliser maps "kal", "kl" and "kall" to "tomorrow".
5. **Confidence gate:** below 0.35 the agent never answers from policy. It asks again, and hands off after two misses.
6. **Tight context budget** (~600 chars): it gets read aloud, and a shorter context makes the LLM faster.
7. **LRU cache**, because delivery calls repeat the same questions.
8. **Citations on every answer**, kept in the audit trail.

Current eval (26 hand-written queries, off-topic set of 6): top-1 81%, recall@3 96%, MRR 0.865, 6/6 off-topic rejected, p50 0.54 ms cold and 0.06 ms cached. These numbers are optimistic because the questions were written alongside the policies. Rebuild the golden set from real pilot transcripts before trusting them.

To swap in a real multilingual embedding model, pass `embed_fn` to `RagPipeline`.

## What this MVP does not do yet

- **Real telephony.** It needs Exotel, Plivo or Twilio SIP streaming, plus streaming STT/TTS such as Sarvam, Deepgram or ElevenLabs. The browser's Web Speech API stands in for them in the demo.
- **An audio emotion model.** Prosody here is a rough loudness and variability proxy.
- **A real OMS or courier-aggregator integration.** `MockOMS` defines the interface, and `/api/webhooks/ndr` accepts NDR events.
- **Multi-tenant auth, persistence of call sessions, queueing and retries.**

All demo orders are synthetic.
