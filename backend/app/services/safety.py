"""Medical safety checks, emergency guidance, and uncertainty handling."""

from __future__ import annotations

import re
from typing import Any

from app.schemas import EMERGENCY_MESSAGE, HEALTHCARE_DISCLAIMER

EMERGENCY_PATTERNS = [
    r"\bchest pain\b",
    r"\bcan'?t breathe\b",
    r"\bcannot breathe\b",
    r"\bdifficulty breathing\b",
    r"\bshortness of breath\b",
    r"\bsuicid",
    r"\bkill myself\b",
    r"\boverdose\b",
    r"\bstroke\b",
    r"\bheart attack\b",
    r"\bsevere bleeding\b",
    r"\bunconscious\b",
    r"\bnot breathing\b",
    r"\banaphylaxis\b",
    r"\bseizure\b",
    r"\bpoison",
]

HARMFUL_PATTERNS = [
    r"\bhow to (make|synthesize|produce) (a )?(poison|toxin|bioweapon)\b",
    r"\bhow to (commit|perform) suicide\b",
]

UNCERTAINTY_CUES = [
    "i'm not sure",
    "i am not sure",
    "uncertain",
    "cannot determine",
    "insufficient",
    "consult a doctor",
    "seek medical",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in patterns)


def analyze_user_message(message: str) -> dict[str, Any]:
    """Return safety flags for a user message."""
    text = message.strip()
    is_emergency = _matches_any(text, EMERGENCY_PATTERNS)
    is_harmful = _matches_any(text, HARMFUL_PATTERNS)
    return {
        "is_emergency": is_emergency,
        "is_harmful": is_harmful,
        "block_generation": is_emergency or is_harmful,
    }


def emergency_response() -> str:
    return EMERGENCY_MESSAGE


def harmful_response() -> str:
    return (
        "I cannot help with that request. "
        "If you are in crisis, please contact local emergency services or a crisis hotline. "
        "For medical questions, please consult a licensed clinician."
    )


def wrap_answer(answer: str, *, uncertain: bool = False) -> str:
    """Ensure educational framing and uncertainty language when needed."""
    text = answer.strip()
    if uncertain and not any(c in text.lower() for c in UNCERTAINTY_CUES):
        text += (
            "\n\nNote: This information may be incomplete. "
            "Please consult a qualified healthcare professional for advice specific to your situation."
        )
    return text


def disclaimer() -> str:
    return HEALTHCARE_DISCLAIMER
