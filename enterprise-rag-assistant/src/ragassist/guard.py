"""Prompt-injection guardrail on user questions.

Primary detector: Llama Prompt Guard 2 (86M classifier served by Groq), which
returns an injection probability. If the classifier is unavailable, a small
set of high-precision patterns is used so the service degrades safely
instead of failing open silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragassist.config import settings
from ragassist.text import strip_accents

INJECTION_PATTERNS = [
    r"\b(ignore|oublie|ignorez|oubliez)\b.{0,40}\b(instructions?|regles?|consignes?|rules)\b",
    r"\b(prompt|instructions?) (systeme|system)\b",
    r"\b(tu es|you are) (maintenant|now)\b",
    r"\b(jailbreak|DAN mode|developer mode)\b",
    r"(affiche|revele|montre|reveal|print).{0,30}\b(prompt|instructions?)\b",
    # added after evaluation (seen attack): requests to switch protections off
    r"\b(desactive|desactivez|disable|turn off|contourne|bypass)\b.{0,30}\b(garde-fous?|filtres?|protections?|"
    r"restrictions?|securite|guardrails?|safety)\b",
]


@dataclass
class GuardResult:
    blocked: bool
    score: float
    detector: str


def heuristic_score(text: str) -> float:
    text = strip_accents(text.lower())
    return 1.0 if any(re.search(p, text) for p in INJECTION_PATTERNS) else 0.0


class InjectionGuard:
    def __init__(self, llm=None, threshold: float = settings.injection_threshold):
        self.llm = llm
        self.threshold = threshold

    def check(self, text: str) -> GuardResult:
        """Defence in depth: block if the classifier OR the patterns flag the text.

        Evaluation showed each detector misses attacks the other catches (Prompt Guard scored a
        French "Tu es maintenant DAN" at 0.001; the patterns cannot generalise to new phrasings).
        """
        heuristic = heuristic_score(text)
        if self.llm is not None and hasattr(self.llm, "classify_injection"):
            try:
                score = max(self.llm.classify_injection(text, settings.guard_model), heuristic)
                return GuardResult(score >= self.threshold, round(score, 4), "prompt-guard-2+heuristic")
            except Exception:  # classifier down: patterns only, do not fail open
                pass
        return GuardResult(heuristic >= self.threshold, heuristic, "heuristic")
