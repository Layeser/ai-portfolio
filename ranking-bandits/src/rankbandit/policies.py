"""Ranking policies for the cascade model (Kveton et al., 2015 "Cascading Bandits").

All policies share the cascade update: positions above the click are observed as 0, the clicked
position as 1, positions below the click are left unobserved.
"""

from __future__ import annotations

import numpy as np


class Policy:
    name = "policy"

    def __init__(self, n_items: int, k: int, rng: np.random.Generator):
        self.n_items, self.k, self.rng = n_items, k, rng
        self.successes = np.zeros(n_items)
        self.trials = np.zeros(n_items)
        self.t = 0

    def scores(self) -> np.ndarray:
        raise NotImplementedError

    def rank(self) -> np.ndarray:
        """Top-k items by score, best first. Untried items (score inf) come first, in random order."""
        self.t += 1
        scores = self.scores()
        noise = self.rng.random(self.n_items)
        scores = np.where(np.isinf(scores), 1e6 + noise, scores + 1e-9 * noise)  # random tie-break
        top = np.argpartition(-scores, self.k - 1)[: self.k]
        return top[np.argsort(-scores[top])]

    def update(self, ranking: np.ndarray, click: int | None) -> None:
        last_seen = click if click is not None else len(ranking) - 1
        seen = ranking[: last_seen + 1]
        self.trials[seen] += 1
        if click is not None:
            self.successes[ranking[click]] += 1

    @property
    def means(self) -> np.ndarray:
        return self.successes / np.maximum(self.trials, 1)


class RandomRanking(Policy):
    name = "Random"

    def scores(self) -> np.ndarray:
        return self.rng.random(self.n_items)


class EpsilonGreedy(Policy):
    name = "ε-greedy (ε=0.05)"

    def __init__(self, n_items, k, rng, epsilon: float = 0.05):
        super().__init__(n_items, k, rng)
        self.epsilon = epsilon

    def scores(self) -> np.ndarray:
        if self.rng.random() < self.epsilon:
            return self.rng.random(self.n_items)
        return np.where(self.trials == 0, np.inf, self.means)  # try every item once first


class CascadeUCB1(Policy):
    name = "CascadeUCB1"

    def scores(self) -> np.ndarray:
        bonus = np.sqrt(1.5 * np.log(self.t + 1) / np.maximum(self.trials, 1))
        return np.where(self.trials == 0, np.inf, np.minimum(self.means + bonus, 1.0))


class CascadeKLUCB(Policy):
    """Tighter optimistic index for Bernoulli rewards: max q s.t. N * KL(mean, q) <= log t."""

    name = "CascadeKL-UCB"

    def scores(self) -> np.ndarray:
        mean, n = np.clip(self.means, 1e-6, 1 - 1e-6), np.maximum(self.trials, 1)
        budget = np.log(self.t + 1) / n
        low, high = mean.copy(), np.ones_like(mean)
        for _ in range(20):  # bisection on q
            q = (low + high) / 2
            kl = mean * np.log(mean / q) + (1 - mean) * np.log((1 - mean) / np.clip(1 - q, 1e-9, None))
            too_far = kl > budget
            high = np.where(too_far, q, high)
            low = np.where(too_far, low, q)
        return np.where(self.trials == 0, np.inf, low)


class CascadeThompson(Policy):
    """Beta(1 + clicks, 1 + seen-without-click) posterior per item; rank by a posterior sample."""

    name = "Cascade Thompson"

    def scores(self) -> np.ndarray:
        return self.rng.beta(1 + self.successes, 1 + self.trials - self.successes)


class DiscountedThompson(CascadeThompson):
    """Thompson sampling that forgets old evidence (factor gamma per round): tracks popularity shifts."""

    name = "Discounted Thompson (γ=0.999)"

    def __init__(self, n_items, k, rng, gamma: float = 0.999):
        super().__init__(n_items, k, rng)
        self.gamma = gamma

    def update(self, ranking, click):
        self.successes *= self.gamma
        self.trials *= self.gamma
        super().update(ranking, click)


ALL_POLICIES = [RandomRanking, EpsilonGreedy, CascadeUCB1, CascadeKLUCB, CascadeThompson, DiscountedThompson]
