"""Emotion and sentiment layer.

Two jobs:
1. Detect how the caller feels, per turn and as a trend across the call,
   from their words (English, romanised Hindi, Devanagari) and, when the
   telephony layer supplies them, simple prosody features.
2. Turn that into a ResponseStyle the agent must follow: what to
   acknowledge first, how fast and how warmly the TTS speaks, how short the
   reply is, and whether to offer a human.

The text model is a transparent lexicon-and-rules scorer so it runs with no
model download and every decision is explainable in the audit log. The
`EmotionDetector.classifier` hook lets you drop in a trained model (e.g. a
fine-tuned IndicBERT or an audio emotion model) later; its output is fused
with the rules rather than replacing them.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

EMOTIONS = ("angry", "frustrated", "anxious", "confused", "happy", "neutral")

LEXICON: dict[str, dict[str, float]] = {
    "angry": {
        "angry": 0.8, "furious": 1.0, "nonsense": 0.8, "bakwas": 0.9, "bakwaas": 0.9,
        "ghatiya": 0.9, "worst": 0.8, "useless": 0.7, "bekar": 0.7, "bekaar": 0.7,
        "fraud": 0.9, "cheat": 0.9, "cheating": 0.9, "chor": 0.9, "pagal": 0.8,
        "stupid": 0.8, "idiot": 0.9, "shut": 0.6, "complaint": 0.5, "consumer": 0.6,
        "court": 0.8, "legal": 0.7, "police": 0.8, "hate": 0.8, "disgusting": 0.9,
        "गुस्सा": 0.8, "बकवास": 0.9, "घटिया": 0.9, "धोखा": 0.9, "बेकार": 0.7,
        "gussa": 0.8, "dhokha": 0.9, "shame": 0.6, "ridiculous": 0.8,
    },
    "frustrated": {
        "again": 0.4, "already": 0.4, "still": 0.4, "times": 0.3, "baar": 0.4,
        "phir": 0.4, "fir": 0.4, "dobara": 0.3, "tired": 0.6, "fed": 0.6,
        "waiting": 0.5, "wait": 0.3, "late": 0.5, "delay": 0.6, "delayed": 0.6,
        "never": 0.5, "annoying": 0.7, "pareshan": 0.7, "pareshaan": 0.7,
        "problem": 0.4, "issue": 0.3, "kitni": 0.4, "kab": 0.2, "tang": 0.7,
        "परेशान": 0.7, "फिर": 0.4, "disappointed": 0.7, "irritated": 0.7,
    },
    "anxious": {
        "worried": 0.7, "urgent": 0.7, "urgently": 0.7, "important": 0.4, "please": 0.2,
        "need": 0.3, "tension": 0.7, "chinta": 0.7, "jaldi": 0.5, "asap": 0.6,
        "gift": 0.4, "birthday": 0.4, "wedding": 0.5, "shaadi": 0.5, "medicine": 0.7,
        "dawai": 0.7, "scared": 0.7, "lost": 0.5, "safe": 0.3, "चिंता": 0.7, "जल्दी": 0.5,
    },
    "confused": {
        "confused": 0.8, "understand": 0.4, "samajh": 0.5, "samjha": 0.5, "matlab": 0.5,
        "what": 0.2, "kya": 0.2, "huh": 0.6, "sorry": 0.2, "repeat": 0.5,
        "which": 0.3, "kaunsa": 0.4, "kaun": 0.3, "didn't": 0.2, "unclear": 0.7,
        "समझ": 0.5, "मतलब": 0.5,
    },
    "happy": {
        "thanks": 0.6, "thank": 0.6, "great": 0.7, "good": 0.5, "perfect": 0.8,
        "awesome": 0.8, "nice": 0.5, "happy": 0.7, "shukriya": 0.7, "dhanyavaad": 0.7,
        "dhanyawad": 0.7, "badhiya": 0.7, "accha": 0.4, "achha": 0.4, "mast": 0.6,
        "khush": 0.7, "helpful": 0.7, "sure": 0.3, "ok": 0.2, "theek": 0.2,
        "धन्यवाद": 0.7, "शुक्रिया": 0.7, "बढ़िया": 0.7, "अच्छा": 0.4,
    },
}

INTENSIFIERS = {"very", "so", "too", "really", "extremely", "bahut", "bohot", "bht",
                "bilkul", "ekdum", "itna", "kitna", "totally", "बहुत", "एकदम"}
NEGATORS = {"not", "no", "never", "nahi", "nahin", "nai", "mat", "na", "नहीं", "मत"}
THREAT_WORDS = {"court", "legal", "police", "consumer", "fraud", "cheat", "cheating", "dhokha", "धोखा"}
_TOKEN = re.compile(r"[A-Za-z']+|[ऀ-ॿ]+")


@dataclass
class EmotionState:
    label: str = "neutral"
    intensity: float = 0.0          # 0..1
    sentiment: float = 0.0          # -1..1
    scores: dict = field(default_factory=dict)
    cues: list = field(default_factory=list)
    trend: str = "stable"           # escalating | calming | stable

    def to_dict(self):
        return asdict(self)


@dataclass
class ResponseStyle:
    tone: str
    acknowledgement: str            # spoken before the task content
    tts_rate: float                 # 1.0 = normal
    tts_pitch: float
    max_sentences: int
    apologise: bool
    offer_human: bool
    escalate_now: bool
    reason: str

    def to_dict(self):
        return asdict(self)


class EmotionDetector:
    def __init__(self, classifier=None):
        # classifier(text) -> {emotion: prob}; optional, fused 50/50 with rules
        self.classifier = classifier

    def detect(self, text: str, prosody: dict | None = None,
               history: list[EmotionState] | None = None) -> EmotionState:
        toks = _TOKEN.findall(text)
        low = [t.lower() for t in toks]
        scores = {e: 0.0 for e in EMOTIONS}
        cues: list[str] = []

        for i, t in enumerate(low):
            window = low[max(0, i - 3):i]
            boost = 1.5 if any(w in INTENSIFIERS for w in window) else 1.0
            negated = any(w in NEGATORS for w in window) or (
                i + 1 < len(low) and low[i + 1] in NEGATORS)  # Hindi puts negation after
            for emo, lex in LEXICON.items():
                if t in lex:
                    w = lex[t] * boost
                    if negated and emo == "happy":
                        scores["frustrated"] += w * 0.8
                        cues.append(f"negated:{t}")
                    else:
                        scores[emo] += w
                        cues.append(f"{emo}:{t}")

        # Surface cues typical of an upset caller on a transcript
        caps = [t for t in toks if len(t) > 2 and t.isupper()]
        if caps:
            scores["angry"] += 0.3 * len(caps)
            cues.append("caps")
        bangs = text.count("!")
        if bangs:
            scores["angry"] += min(0.6, 0.2 * bangs)
            cues.append("exclamation")
        if text.count("?") >= 2:
            scores["confused"] += 0.3
            cues.append("multi_question")
        repeats = len(low) - len(set(low))
        if len(low) > 4 and repeats / len(low) > 0.3:
            scores["frustrated"] += 0.3
            cues.append("repetition")

        if prosody:
            energy = float(prosody.get("energy", 0.5))
            pitch_var = float(prosody.get("pitch_var", 0.5))
            rate = float(prosody.get("speech_rate", 2.5))   # words/sec
            if energy > 0.75 and pitch_var > 0.6:
                scores["angry"] += 0.5
                cues.append("prosody:loud_variable")
            elif rate > 3.5:
                scores["anxious"] += 0.4
                cues.append("prosody:fast")
            elif energy < 0.25 and rate < 1.5:
                scores["frustrated"] += 0.2
                cues.append("prosody:flat")

        if self.classifier:
            for emo, p in self.classifier(text).items():
                if emo in scores:
                    scores[emo] = 0.5 * scores[emo] + 0.5 * p * 2

        label = max(scores, key=scores.get)
        top = scores[label]
        if top < 0.35:
            label, top = "neutral", 0.0
        intensity = round(min(1.0, top / 1.6), 2)
        neg = scores["angry"] + scores["frustrated"] + 0.6 * scores["anxious"]
        pos = scores["happy"]
        sentiment = round((pos - neg) / (pos + neg + 1.0), 2)

        state = EmotionState(label, intensity, sentiment,
                             {k: round(v, 2) for k, v in scores.items()}, cues)
        state.trend = self._trend(state, history or [])
        return state

    @staticmethod
    def _trend(cur: EmotionState, history: list[EmotionState]) -> str:
        if not history:
            return "stable"
        prev = history[-1].sentiment
        if cur.sentiment < prev - 0.15:
            return "escalating"
        if cur.sentiment > prev + 0.15:
            return "calming"
        return "stable"


ACK = {
    "angry": {
        "en": ["I completely understand why you are upset, and I am sorry about this.",
               "I hear you, and I am really sorry for the trouble."],
        "hi": ["Main aapki pareshani samajh sakti hoon, iske liye hum maafi chahte hain.",
               "Aapki baat bilkul sahi hai, is taklif ke liye sorry."],
    },
    "frustrated": {
        "en": ["I am sorry you have had to deal with this more than once.",
               "I understand this has been frustrating."],
        "hi": ["Sorry, aapko baar baar pareshan hona pada.",
               "Main samajhti hoon, yeh kaafi frustrating raha hoga."],
    },
    "anxious": {
        "en": ["Don't worry, I will help you sort this out right now.",
               "I understand this is important to you. Let us fix it quickly."],
        "hi": ["Aap chinta mat kijiye, main abhi isse theek karti hoon.",
               "Samajh sakti hoon yeh zaroori hai, chaliye jaldi karte hain."],
    },
    "confused": {
        "en": ["No problem, let me explain that more simply.", "Sure, let me put it another way."],
        "hi": ["Koi baat nahi, main aasaan shabdon mein batati hoon.",
               "Zaroor, main dobara samjhaati hoon."],
    },
    "happy": {
        "en": ["Wonderful!", "Great to hear that!"],
        "hi": ["Bahut badhiya!", "Yeh sunkar accha laga!"],
    },
    "neutral": {"en": [""], "hi": [""]},
}


def style_for(state: EmotionState, history: list[EmotionState], lang: str = "en",
              turn: int = 0) -> ResponseStyle:
    """Map the detected emotion (and its trend) to how the agent must speak.

    Escalation rule mirrors escalation_policy.md: very angry once, or angry
    in two consecutive turns, triggers a handoff offer."""
    key = "hi" if lang.startswith("hi") else "en"
    lines = ACK[state.label][key]
    ack = lines[turn % len(lines)]
    prev_angry = bool(history) and history[-1].label == "angry"
    very_angry = state.label == "angry" and state.intensity >= 0.8
    two_in_row = state.label == "angry" and prev_angry
    threat = any(c.split(":")[-1] in THREAT_WORDS for c in state.cues)
    # Escalate at once on a threat or sustained anger. A single very angry
    # turn with an actionable request is handled with empathy first; the
    # agent escalates that case itself if it cannot act (see agent._route).
    escalate = threat or two_in_row

    if state.label == "angry":
        return ResponseStyle("calm, slow, apologetic", ack, 0.9, 0.95, 2, True, True,
                             escalate, "legal or fraud threat" if threat else
                             ("angry two turns in a row" if two_in_row else
                              ("very angry" if very_angry else "angry")))
    if state.label == "frustrated":
        return ResponseStyle("patient, reassuring", ack, 0.95, 1.0, 2, True,
                             state.trend == "escalating", False, "frustrated")
    if state.label == "anxious":
        return ResponseStyle("reassuring, brisk", ack, 1.0, 1.0, 2, False, False, False, "anxious")
    if state.label == "confused":
        return ResponseStyle("simple, slow, one step at a time", ack, 0.85, 1.0, 1, False,
                             False, False, "confused")
    if state.label == "happy":
        return ResponseStyle("warm, upbeat", ack, 1.05, 1.05, 2, False, False, False, "happy")
    return ResponseStyle("friendly, clear", "", 1.0, 1.0, 2, False, False, False, "neutral")
