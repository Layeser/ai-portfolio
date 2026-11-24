"""Official M5 accuracy metric: WRMSSE over 12 aggregation levels (42,840 series).

RMSSE of a series = RMSE of the forecast / RMSE of the one-step naive forecast on its own history
(from its first non-zero sale). Each level weighs its series by their dollar sales over the last 28
days of history; the 12 levels are averaged. Reference: Makridakis et al., "The M5 accuracy
competition" (IJF 2022). Published scores on the evaluation period: sNaive 0.847, top-1 0.520.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from salesfc.data import HORIZON, price_matrix, sales_wide

LEVELS = [  # groupings of the 30,490 bottom series, in the competition's order
    [], ["state_id"], ["store_id"], ["cat_id"], ["dept_id"], ["state_id", "cat_id"], ["state_id", "dept_id"],
    ["store_id", "cat_id"], ["store_id", "dept_id"], ["item_id"], ["item_id", "state_id"], ["item_id", "store_id"],
]


class WRMSSE:
    """Precomputes aggregation matrices, scales and weights for a forecast starting at day `start` (1-based)."""

    def __init__(self, start: int):
        ids, sales = sales_wide()
        hist = sales[:, : start - 1]
        dollars = (hist[:, -HORIZON:] * np.nan_to_num(price_matrix()[:, start - 1 - HORIZON : start - 1])).sum(1)
        self.truth = sales[:, start - 1 : start - 1 + HORIZON]
        self.levels = []
        for cols in LEVELS:
            key = ids[cols].astype(str).agg("_".join, axis=1) if cols else pd.Series("total", index=ids.index)
            codes, uniques = pd.factorize(key)
            # sparse aggregation matrix: the bottom level alone would be a dense 30,490 x 30,490 (3.7 GB)
            onehot = sp.csr_matrix((np.ones(len(ids), dtype=np.float32), (codes, np.arange(len(ids)))),
                                   shape=(len(uniques), len(ids)))
            agg_hist = np.asarray(onehot @ hist)
            first = np.where((agg_hist > 0).any(1), (agg_hist > 0).argmax(1), agg_hist.shape[1] - 1)
            diffs = np.diff(agg_hist, axis=1) ** 2
            mask = np.arange(1, agg_hist.shape[1])[None] > first[:, None]
            scale = (diffs * mask).sum(1) / np.maximum(mask.sum(1), 1)
            weight = np.asarray(onehot @ dollars).ravel()
            self.levels.append({"onehot": onehot, "scale": np.maximum(scale, 1e-8), "weight": weight / weight.sum()})

    def __call__(self, forecast: np.ndarray, per_level: bool = False):
        assert forecast.shape == self.truth.shape
        scores = []
        for lv in self.levels:
            err = ((lv["onehot"] @ forecast - lv["onehot"] @ self.truth) ** 2).mean(1)
            scores.append(float((lv["weight"] * np.sqrt(err / lv["scale"])).sum()))
        total = float(np.mean(scores))
        return (total, scores) if per_level else total
