"""RAG evaluation: retrieval quality, off-topic rejection and latency.

    python -m tests.eval_rag

Golden set is small and hand-written; grow it from real call transcripts
(the queries that ended in low confidence or escalation) during the pilot.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

from voxara.nlu import parse
from voxara.rag import RagPipeline

KB = Path(__file__).resolve().parents[1] / "voxara" / "kb"

# (caller utterance, workflow, expected chunk id)
GOLDEN = [
    ("kal dobara bhej do", "ndr_recovery", "ndr_policy#reattempt-rules"),
    ("can you deliver next week", "ndr_recovery", "ndr_policy#reattempt-rules"),
    ("delivery boy ne call hi nahi kiya", "ndr_recovery", "ndr_policy#fake-attempt-complaints"),
    ("the courier never came to my house", "ndr_recovery", "ndr_policy#fake-attempt-complaints"),
    ("maine order refuse kiya tha kyunki late tha", "ndr_recovery", "ndr_policy#refusal-handling"),
    ("I rejected the parcel", "ndr_recovery", "ndr_policy#refusal-handling"),
    ("I want to talk to a manager", "ndr_recovery", "escalation_policy#when-the-agent-hands-off-to-a-human"),
    ("kisi insaan se baat karao", "ndr_recovery", "escalation_policy#when-the-agent-hands-off-to-a-human"),
    ("will someone call me back", "ndr_recovery", "escalation_policy#callback"),
    ("mujhe online pay karna hai", "cod_confirmation", "cod_policy#switching-to-prepaid"),
    ("can I pay by UPI instead", "cod_confirmation", "cod_policy#switching-to-prepaid"),
    ("itne zyada paise kyun", "cod_confirmation", "cod_policy#amount-questions"),
    ("why is the amount so high", "cod_confirmation", "cod_policy#amount-questions"),
    ("I did not place this order, cancel it", "cod_confirmation", "cod_policy#cancellation"),
    ("is there a cancellation fee", "cod_confirmation", "cod_policy#cancellation"),
    ("mera address badalna hai", "ndr_recovery", "faq_hinglish#address-badalna-hai"),
    ("the landmark is near the temple", "address_fix", "address_policy#landmark"),
    ("change pincode to another city", "address_fix", "address_policy#pincode-change"),
    ("deliver to my alternate phone number", "address_fix", "address_policy#alternate-phone-number"),
    ("what time will it come exactly", "delivery_scheduling", "reschedule_policy#delivery-slots"),
    ("can you deliver today itself", "delivery_scheduling", "reschedule_policy#how-far-ahead"),
    ("can I pick it up from the hub", "delivery_scheduling", "reschedule_policy#hold-at-hub"),
    ("is diwali a delivery day", "delivery_scheduling", "reschedule_policy#holidays"),
    ("please don't record this call", "ndr_recovery", "compliance_policy#recording-and-consent"),
    ("stop calling me so many times", "ndr_recovery", "compliance_policy#attempts"),
    ("refund kab milega", "ndr_recovery", "faq_hinglish#refund-kab-milega"),
]

OFF_TOPIC = ["who is the PM", "what's the weather", "tell me a joke", "cricket score batao",
             "book a movie ticket", "what is your name"]


def main():
    rag = RagPipeline(KB)
    hit1 = hit3 = 0
    rr = []
    lat = []
    misses = []
    for q, wf, gold in GOLDEN:
        intent = parse(q).intent
        t = time.perf_counter()
        r = rag.retrieve(q, intent, wf)
        lat.append((time.perf_counter() - t) * 1000)
        ids = [h["id"] for h in r.hits]
        if ids[:1] == [gold]:
            hit1 += 1
        if gold in ids[:3]:
            hit3 += 1
            rr.append(1 / (ids.index(gold) + 1))
        else:
            rr.append(0)
            misses.append((q, intent, ids[:3]))
    rag._cache.clear()
    rejected = sum(not rag.retrieve(q, "unknown", "ndr_recovery").grounded for q in OFF_TOPIC)
    grounded_gold = sum(rag.retrieve(q, parse(q).intent, wf).grounded for q, wf, _ in GOLDEN)

    # Cached latency: second pass over the same questions
    cached = []
    for q, wf, _ in GOLDEN:
        t = time.perf_counter()
        rag.retrieve(q, parse(q).intent, wf)
        cached.append((time.perf_counter() - t) * 1000)

    n = len(GOLDEN)
    print(f"Golden queries          : {n}")
    print(f"Top-1 accuracy          : {hit1 / n:.0%}")
    print(f"Recall@3                : {hit3 / n:.0%}")
    print(f"MRR@3                   : {sum(rr) / n:.3f}")
    print(f"Answerable & grounded   : {grounded_gold / n:.0%}  (confidence >= {rag.CONFIDENCE_THRESHOLD})")
    print(f"Off-topic rejected      : {rejected}/{len(OFF_TOPIC)}")
    print(f"Latency p50 / p95 (cold): {statistics.median(lat):.2f} / {sorted(lat)[int(0.95 * n) - 1]:.2f} ms")
    print(f"Latency p50 (cached)    : {statistics.median(cached):.3f} ms")
    for m in misses:
        print("  MISS:", m)


if __name__ == "__main__":
    main()
