from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def ensure_output_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def detect_label_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No label column found. Expected one of: {list(candidates)}")


def normalize_label(value: object) -> str:
    text = str(value).strip().replace(" ", "")
    if "-" in text:
        text = text.split("-")[-1]
    if text.upper() in {"SYN", "SYNFLOOD", "SYN_FLOOD"}:
        return "Syn"
    return text


def cap_class_samples(df: pd.DataFrame, label_col: str, max_samples: int, random_state: int) -> pd.DataFrame:
    return (
        df.groupby(label_col, group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), max_samples), random_state=random_state))
        .reset_index(drop=True)
    )


def keep_numeric_features(df: pd.DataFrame, label_col: str, drop_cols: Iterable[str]) -> pd.DataFrame:
    to_drop = [c for c in drop_cols if c in df.columns and c != label_col]
    clean = df.drop(columns=to_drop, errors="ignore")
    numeric_cols = clean.select_dtypes(include=[np.number]).columns.tolist()
    if label_col not in numeric_cols and label_col in clean.columns:
        numeric_cols.append(label_col)
    return clean[numeric_cols].copy()


def remove_correlated_features(df: pd.DataFrame, threshold: float, ignore_cols: Iterable[str]) -> tuple[pd.DataFrame, list[str]]:
    ignore = set(ignore_cols)
    work = df.drop(columns=[c for c in ignore if c in df.columns], errors="ignore")
    corr = work.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > threshold)]
    reduced = df.drop(columns=drop_cols, errors="ignore")
    return reduced, drop_cols


def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to 1")

    train_df, tmp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=df[label_col],
        random_state=random_state,
    )
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        tmp_df,
        test_size=relative_test_size,
        stratify=tmp_df[label_col],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
