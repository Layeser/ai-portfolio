"""Online ranking service: serve a top-k list, learn from the click (or the absence of click).

    GET  /ranking            -> {"ranking_id", "items"}    (Cascade Thompson sampling)
    POST /feedback           <- {"ranking_id", "clicked_position": int | null}
    GET  /stats              -> best items so far, impressions, clicks

Rankings must be answered within `FEEDBACK_TTL` served rankings; older ones are rejected, so a
stale or replayed callback cannot corrupt the posterior. State is in memory and checkpointed to
JSON so a restart keeps what was learned.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rankbandit.policies import CascadeThompson

N_ITEMS = int(os.getenv("RANKBANDIT_ITEMS", "100"))
K = int(os.getenv("RANKBANDIT_K", "5"))
STATE = Path(os.getenv("RANKBANDIT_STATE", "rankbandit_state.json"))
FEEDBACK_TTL = 10_000


class Ranking(BaseModel):
    ranking_id: str
    items: list[int]


class Feedback(BaseModel):
    ranking_id: str
    clicked_position: int | None = Field(None, ge=0, description="0-based position of the click, null if none")


class Service:
    def __init__(self, n_items: int = N_ITEMS, k: int = K, state: Path | None = STATE, seed: int | None = None):
        self.policy = CascadeThompson(n_items, k, np.random.default_rng(seed))
        self.pending: OrderedDict[str, np.ndarray] = OrderedDict()
        self.lock = threading.Lock()
        self.state = state
        self.impressions = self.clicks = 0
        if state is not None and state.exists():
            saved = json.loads(state.read_text())
            self.policy.successes = np.array(saved["successes"])
            self.policy.trials = np.array(saved["trials"])
            self.impressions, self.clicks = saved["impressions"], saved["clicks"]

    def serve(self) -> Ranking:
        with self.lock:
            items = self.policy.rank()
            ranking_id = uuid.uuid4().hex
            self.pending[ranking_id] = items
            if len(self.pending) > FEEDBACK_TTL:
                self.pending.popitem(last=False)
            self.impressions += 1
        return Ranking(ranking_id=ranking_id, items=items.tolist())

    def feedback(self, fb: Feedback) -> None:
        with self.lock:
            items = self.pending.pop(fb.ranking_id, None)
            if items is None:
                raise KeyError(fb.ranking_id)
            if fb.clicked_position is not None and fb.clicked_position >= len(items):
                self.pending[fb.ranking_id] = items  # keep it: the caller can retry with a valid position
                raise ValueError("clicked_position outside the ranking")
            self.policy.update(items, fb.clicked_position)
            self.clicks += fb.clicked_position is not None
            if self.state is not None and self.impressions % 100 == 0:
                self.save()

    def save(self) -> None:
        self.state.write_text(
            json.dumps(
                {
                    "successes": self.policy.successes.tolist(),
                    "trials": self.policy.trials.tolist(),
                    "impressions": self.impressions,
                    "clicks": self.clicks,
                }
            )
        )

    def stats(self) -> dict:
        means = self.policy.means
        top = np.argsort(-means)[:K]
        return {
            "impressions": self.impressions,
            "clicks": self.clicks,
            "ctr": round(self.clicks / max(self.impressions, 1), 4),
            "best_items": [
                {
                    "item": int(i),
                    "estimated_attraction": round(float(means[i]), 4),
                    "observations": int(self.policy.trials[i]),
                }
                for i in top
            ],
        }


app = FastAPI(title="Online ranking (cascading bandits)", version="1.0.0")
app.state.service = Service()


@app.get("/ranking", response_model=Ranking)
def ranking() -> Ranking:
    return app.state.service.serve()


@app.post("/feedback", status_code=204)
def feedback(fb: Feedback) -> None:
    try:
        app.state.service.feedback(fb)
    except KeyError as exc:
        raise HTTPException(404, "unknown or expired ranking_id") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/stats")
def stats() -> dict:
    return app.state.service.stats()
