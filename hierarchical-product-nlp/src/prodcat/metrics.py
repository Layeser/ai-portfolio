"""Flat and hierarchical metrics, and the auto-categorisation coverage trade-off."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from prodcat.taxonomy import Taxonomy


def hierarchical_prf(y_true: list[str], y_pred: list[str], tax: Taxonomy) -> dict:
    """Kiritchenko et al. hierarchical P/R/F1 on ancestor sets (shared root excluded).

    A wrong leaf that is a sibling of the right one still gets partial credit, which matches
    the business cost: "LED TV" instead of "OLED TV" is a much smaller error than "garden hose".
    """
    inter = n_pred = n_true = 0
    for t, p in zip(y_true, y_pred, strict=True):
        a_true, a_pred = set(tax.path(t)[1:]), set(tax.path(p)[1:])
        inter += len(a_true & a_pred)
        n_pred += len(a_pred)
        n_true += len(a_true)
    hp, hr = inter / n_pred, inter / n_true
    hf1 = 2 * hp * hr / (hp + hr) if hp + hr else 0.0  # disjoint paths: no shared ancestor at all
    return {"hP": round(hp, 4), "hR": round(hr, 4), "hF1": round(hf1, 4)}


def level_accuracy(y_true: list[str], y_pred: list[str], tax: Taxonomy, max_level: int = 4) -> dict:
    """Accuracy of the predicted path truncated at each depth (level 1 = first split under the root)."""
    out = {}
    for level in range(1, max_level + 1):
        ok = [tax.path(t)[: level + 1] == tax.path(p)[: level + 1] for t, p in zip(y_true, y_pred, strict=True)]
        out[f"level_{level}"] = round(float(np.mean(ok)), 4)
    return out


def classification_report(y_true: list[str], y_pred: list[str], tax: Taxonomy) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        **hierarchical_prf(y_true, y_pred, tax),
        "level_accuracy": level_accuracy(y_true, y_pred, tax),
    }


def coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(thresholds, coverage, accuracy on covered items) when auto-accepting confidence >= threshold."""
    order = np.argsort(-confidence)
    conf, ok = confidence[order], correct[order].astype(float)
    covered = np.arange(1, len(conf) + 1)
    return conf, covered / len(conf), np.cumsum(ok) / covered


def threshold_for_accuracy(confidence: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Lowest confidence threshold whose auto-accepted predictions reach `target` accuracy."""
    thresholds, _, acc = coverage_curve(confidence, correct)
    ok = np.where(acc >= target)[0]
    return float(thresholds[ok[-1]]) if len(ok) else 1.0
