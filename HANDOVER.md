# Handover: continuing Voxara Deliver

Working MVP built on 25 September 2026. To run it locally:

```bash
git clone https://github.com/ankit17kumar/voxara-deliver && cd voxara-deliver
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q          # expect 35 passed
python -m tests.eval_rag     # retrieval accuracy and latency
uvicorn voxara.api:app --reload   # open http://localhost:8000 in Chrome
```

## Design decisions already made (and why)

1. **Rules first, LLM on the long tail.** Intents, slots and workflows are deterministic, so they're free, instant and auditable. An LLM only rephrases grounded policy answers and classifies what the rules miss (`voxara/llm.py`, active when `ANTHROPIC_API_KEY` is set).
2. **Order facts never come from RAG.** They come from the OMS. RAG only answers policy questions, and it refuses below a confidence of 0.35.
3. **Emotion changes how the agent speaks, not what it's allowed to do.** Compliance and workflow rules are hard code that no emotion or model output can override.
4. **Angry but fixable means empathy plus the fix, not a transfer.** It escalates at once only on a legal or fraud threat, on anger two turns running, or on very angry callers it can't help. Tune this in `emotion.style_for` and `agent._route` once you have real pilot data.
5. **Every call ends in exactly one outcome code** that is written to the OMS. That is what the KPIs are computed from.

## Suggested next steps, in order

1. **Telephony.** Stream audio from Exotel or Plivo (Indian DIDs, needed for TRAI CLI registration) over WebSocket into `DeliverAgent.handle`.
2. **Streaming STT/TTS for Indian languages.** Try Sarvam and compare it with Deepgram and ElevenLabs on Hinglish word error rate and latency, using 50 real call recordings. Pass real prosody features into `handle(..., prosody=...)`.
3. **One real OMS or aggregator integration** (Shiprocket, ClickPost or the pilot brand's OMS). Implement the `MockOMS` methods against it, and point their NDR webhook at `/api/webhooks/ndr`.
4. **A durable call-session store and a job queue** for retries (Postgres plus a simple worker).
5. **Rebuild `tests/eval_rag.py` from real transcripts.** Start with every turn that ended in low confidence or escalation.
6. **Add regional languages** where RTO is highest: Tamil, Telugu, Kannada, Bengali and Marathi templates in `agent.T`.

## Known limits

- The prosody signal is a rough browser loudness proxy.
- The demo data is synthetic.
- There's no auth or multi-tenancy.
- The Hindi templates are romanised Hinglish written for a female voice persona ("sakti hoon"). Change them if you use a male voice.
