"""Cascade click model: the user scans a ranked list top-down and clicks the first attractive item.

Each item i has an (unknown) attraction probability p_i. Showing the list (a_1, ..., a_K):
    P(click) = 1 - prod_k (1 - p_{a_k})
Feedback is partial: items above the click were seen and *not* clicked, the clicked item was
attractive, items below the click were never examined. If nothing is clicked, all K were seen.

The expected regret of a list is P(click | best K items) - P(click | shown list): it uses the true
probabilities, so curves are smooth and comparable across policies (pseudo-regret).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CascadeEnvironment:
    attraction: np.ndarray  # (L,) true attraction probability of each item
    k: int  # slots in the ranking
    rng: np.random.Generator

    @classmethod
    def catalogue(cls, n_items: int = 100, k: int = 5, seed: int = 0) -> CascadeEnvironment:
        """Realistic CTR profile: most items attract 1-5 % of users, a few stand out (Beta(1, 25) + boosts)."""
        rng = np.random.default_rng(seed)
        p = rng.beta(1.0, 25.0, size=n_items)
        stars = rng.choice(n_items, size=max(1, n_items // 20), replace=False)
        p[stars] = rng.uniform(0.10, 0.25, size=len(stars))
        return cls(np.clip(p, 0.001, 0.99), k, rng)

    @property
    def n_items(self) -> int:
        return len(self.attraction)

    def click_probability(self, ranking: np.ndarray) -> float:
        return float(1.0 - np.prod(1.0 - self.attraction[ranking]))

    @property
    def best_click_probability(self) -> float:
        best = np.argsort(-self.attraction)[: self.k]
        return self.click_probability(best)

    def step(self, ranking: np.ndarray) -> tuple[int | None, float]:
        """Show `ranking`; return (position of the click or None, expected regret of this ranking)."""
        attracted = self.rng.random(len(ranking)) < self.attraction[ranking]
        click = int(np.argmax(attracted)) if attracted.any() else None
        return click, self.best_click_probability - self.click_probability(ranking)

    def shift(self, fraction: float = 0.3) -> None:
        """Non-stationarity: a share of items swap popularity (seasonality, new trends)."""
        n = int(self.n_items * fraction)
        idx = self.rng.choice(self.n_items, size=n, replace=False)
        self.attraction[idx] = self.attraction[self.rng.permutation(idx)]
