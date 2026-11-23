"""Loading, validation and text preparation for product records."""

from __future__ import annotations

import html
import os
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("PRODCAT_DATA_DIR", ROOT / "data"))
TAXONOMY_PATH = ROOT / "tree.json"
REQUIRED = ["product_id", "category_id", "category_path", "title", "description", "brand", "color"]
SEED = 42

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def clean(text) -> str:
    if not isinstance(text, str):
        return ""
    text = _TAG_RE.sub(" ", html.unescape(text))
    return _SPACE_RE.sub(" ", text).strip()


def build_text(df: pd.DataFrame, max_description_chars: int = 600) -> pd.Series:
    """Title first (most informative), then brand / colour, then the start of the description."""
    return (
        df["title"].map(clean) + " | " + df["brand"].map(clean) + " | " + df["color"].map(clean)
        + " | " + df["description"].map(clean).str.slice(0, max_description_chars)
    )


def validate(df: pd.DataFrame, name: str) -> list[str]:
    """Data-quality checks run before training; returns human-readable issues (empty = OK)."""
    issues = [f"{name}: missing column {c}" for c in REQUIRED if c not in df.columns]
    if issues:
        return issues
    if df["product_id"].duplicated().any():
        issues.append(f"{name}: {df['product_id'].duplicated().sum()} duplicated product_id")
    if df["title"].map(clean).eq("").any():
        issues.append(f"{name}: {df['title'].map(clean).eq('').sum()} empty titles")
    bad_paths = (df["category_path"].str.split("/").str[-1] != df["category_id"]).sum()
    if bad_paths:
        issues.append(f"{name}: {bad_paths} rows where category_path does not end with category_id")
    return issues


def load(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{split}set.csv")
    problems = validate(df, split)
    if problems:
        raise ValueError("; ".join(problems))
    df["text"] = build_text(df)
    return df


def train_valid_split(train: pd.DataFrame, valid_size: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(train, test_size=valid_size, stratify=train["category_id"], random_state=SEED)


def leakage_report(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Exact title overlaps between train and test inflate test scores; we report a de-duplicated score too."""
    shared = set(train["title"].map(clean).str.lower()) & set(test["title"].map(clean).str.lower())
    mask = test["title"].map(clean).str.lower().isin(shared)
    return {"test_rows_with_title_seen_in_train": int(mask.sum()), "share": round(float(mask.mean()), 4)}, ~mask
