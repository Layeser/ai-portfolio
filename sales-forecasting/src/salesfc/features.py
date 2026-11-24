"""Feature matrix for global models, built from the three M5 sources (sales, calendar, prices).

Every feature only uses sales at least 28 days old, so one model forecasts the whole 28-day horizon
in a single pass (no recursive forecasting, no error accumulation, no leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from salesfc.data import HORIZON, calendar, first_sale_day, price_matrix, sales_wide

CATEGORICAL = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name_1", "event_type_1"]


def _rolling_mean(m: np.ndarray, cols: np.ndarray, offset: int, window: int) -> np.ndarray:
    """Mean of m[:, c - offset - window + 1 .. c - offset] for every column index c in `cols`."""
    c = np.concatenate([np.zeros((len(m), 1)), np.cumsum(np.nan_to_num(m), axis=1, dtype=np.float64)], 1)
    hi, lo = cols - offset + 1, cols - offset + 1 - window
    return ((c[:, hi] - c[:, np.maximum(lo, 0)]) / window).astype(np.float32)


def build(start: int, train_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows for days [start - train_days, start - 1] and forecast rows for [start, start + 27]
    (1-based M5 days). Returns (train, forecast) frames with features, 'sales' target and 'd'.
    Features are computed only for the needed columns (no full-history matrices)."""
    ids, truth = sales_wide()
    sales = truth[:, : start - 1]  # the future is unknown
    price = price_matrix()
    days = np.arange(start - train_days, start + HORIZON)  # 1-based
    cols = days - 1
    feats = {f"lag_{k}": sales[:, cols - k] for k in (28, 35, 42, 49, 56, 364)}
    for w in (7, 28, 56, 112):
        feats[f"rmean28_{w}"] = _rolling_mean(sales, cols, HORIZON, w)
    sq = _rolling_mean(sales**2, cols, HORIZON, 28)
    feats["rstd28_28"] = np.sqrt(np.maximum(sq - feats["rmean28_28"] ** 2, 0))
    p = price[:, cols]
    feats["price"] = p
    feats["price_rel_max"] = p / np.nanmax(np.where(np.isnan(price), -np.inf, price), axis=1, keepdims=True)
    feats["price_change_7"] = p / price[:, cols - 7]
    dept = ids["dept_id"].astype("category").cat.codes.to_numpy()
    sums = np.zeros((dept.max() + 1, len(cols)))
    counts = np.zeros_like(sums)
    np.add.at(sums, dept, np.nan_to_num(p))
    np.add.at(counts, dept, ~np.isnan(p))
    feats["price_rel_dept"] = p / (sums / np.maximum(counts, 1))[dept]

    first = first_sale_day(sales)
    n_series = len(ids)
    frame = {name: m.astype(np.float32).reshape(-1) for name, m in feats.items()}
    del feats
    frame["sales"] = truth[:, cols].reshape(-1)
    frame["d"] = np.tile(days, n_series)
    frame["age"] = (np.tile(cols, n_series) - np.repeat(first, len(days))).astype(np.float32)
    for c in ID_COLS_CODES:
        frame[c] = np.repeat(ids[c].astype("category").cat.codes.to_numpy(), len(days)).astype(np.int16)
    df = pd.DataFrame(frame)
    del frame
    df["series"] = np.repeat(np.arange(n_series), len(days)).astype(np.int32)

    cal = calendar()
    d = df["d"].to_numpy() - 1  # calendar row index
    df["wday"] = cal["wday"].to_numpy()[d].astype(np.int8)
    df["month"] = cal["month"].to_numpy()[d].astype(np.int8)
    df["mday"] = cal["date"].dt.day.to_numpy()[d].astype(np.int8)
    for c in ("event_name_1", "event_type_1"):
        df[c] = cal[c].astype("category").cat.codes.to_numpy()[d].astype(np.int16)
    state = ids["state_id"].astype("category").cat.codes.to_numpy()[df["series"].to_numpy()]
    snap = cal[["snap_CA", "snap_TX", "snap_WI"]].to_numpy()  # category codes sort CA, TX, WI
    df["snap"] = snap[d, state].astype(np.int8)

    df = df[(df["age"] >= 0) & df["price"].notna()]  # before launch / not on sale: not real zeros
    train, forecast = df[df["d"] < start], df[df["d"] >= start]
    return train.reset_index(drop=True), forecast.reset_index(drop=True)


ID_COLS_CODES = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("sales", "d", "series")]
