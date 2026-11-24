"""Scoring of JSON extractions.

- field F1: each document is flattened to a multiset of (path, value) leaves, e.g. ("menu.nm", "Ice Tea");
  precision / recall are counted over all leaves of the split (micro-average), as in Donut's CORD F1.
- valid JSON rate: share of outputs that parse as a JSON object.
- document accuracy: share of receipts whose every field is exactly right.
Values are compared after whitespace normalisation only, so "75,000" and "75.000" differ.
"""

from __future__ import annotations

import json
import re
from collections import Counter


def parse_json(output: str) -> dict | None:
    """First-to-last brace span of a model output, parsed; None when it is not a JSON object."""
    start, end = output.find("{"), output.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(output[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def leaves(obj, path: str = "") -> Counter:
    out: Counter = Counter()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for v in obj:
            out += leaves(v, path)
    elif obj is not None and str(obj).strip():
        out[(path, re.sub(r"\s+", " ", str(obj)).strip())] += 1
    return out


def score(predictions: list[dict | None], targets: list[dict]) -> dict:
    tp = n_pred = n_true = exact = valid = 0
    per_field: dict[str, Counter] = {}
    for pred, true in zip(predictions, targets, strict=True):
        p, t = leaves(pred or {}), leaves(true)
        hit = p & t
        tp += sum(hit.values())
        n_pred += sum(p.values())
        n_true += sum(t.values())
        exact += p == t
        valid += pred is not None
        for (path, _), n in t.items():
            per_field.setdefault(path, Counter())["true"] += n
        for (path, _), n in hit.items():
            per_field[path]["hit"] += n
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "field_f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4),
        "doc_accuracy": round(exact / len(targets), 4), "valid_json": round(valid / len(targets), 4),
        "n": len(targets),
        "recall_by_field": {k: round(c["hit"] / c["true"], 3) for k, c in sorted(per_field.items(), key=lambda x: -x[1]["true"])},
    }
