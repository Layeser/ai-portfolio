"""M5 (Walmart) data: 30,490 item x store daily sales series, with calendar and price sources.

Periods (M5 "d_" day indices):
* history: d_1 .. d_1941 (2011-01-29 .. 2016-05-22)
* validation fold: forecast d_1914 .. d_1941 from history up to d_1913 (used for every modelling choice)
* test: forecast d_1942 .. d_1969 from history up to d_1941 (the competition's private evaluation period,
  ground truth in sales_test_evaluation.csv; used once, at the end)
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.getenv("SALESFC_DATA", ROOT / "data" / "raw"))
HORIZON = 28
ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
FOLDS = {"validation": 1914, "test": 1942}  # first forecast day of each fold


@cache
def calendar() -> pd.DataFrame:
    cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    cal["d"] = np.arange(1, len(cal) + 1)  # rows are days d_1 .. d_1969 in order
    return cal


@cache
def sales_wide() -> tuple[pd.DataFrame, np.ndarray]:
    """(ids dataframe, sales matrix of shape (30490, 1969)) with test-period truth appended."""
    train = pd.read_csv(RAW / "sales_train_evaluation.csv")
    test = pd.read_csv(RAW / "sales_test_evaluation.csv")
    ids = train[ID_COLS].copy()
    ids["id"] = ids["item_id"] + "_" + ids["store_id"]
    assert (test["item_id"] + "_" + test["store_id"]).equals(ids["id"])
    days = [c for c in train.columns if c.startswith("d_")] + [c for c in test.columns if c.startswith("d_")]
    values = np.concatenate([train.filter(like="d_").to_numpy(), test.filter(like="d_").to_numpy()], 1)
    assert values.shape[1] == len(days) == 1969
    return ids, values.astype(np.float32)


@cache
def prices() -> pd.DataFrame:
    return pd.read_csv(RAW / "sell_prices.csv")


def price_matrix() -> np.ndarray:
    """Sell price of every series on every day (NaN before the item is on sale)."""
    ids, _ = sales_wide()
    cal = calendar()
    p = prices().pivot_table(index=["item_id", "store_id"], columns="wm_yr_wk", values="sell_price")
    p = p.reindex(pd.MultiIndex.from_frame(ids[["item_id", "store_id"]]))
    return p[cal["wm_yr_wk"].to_numpy()].to_numpy(dtype=np.float32)  # (series, 1969)


def first_sale_day(sales: np.ndarray) -> np.ndarray:
    """Index of the first non-zero sale (series are zero-padded before the product launch)."""
    nz = sales > 0
    return np.where(nz.any(1), nz.argmax(1), sales.shape[1])
