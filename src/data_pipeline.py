from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.logging_utils import setup_logger
from src.utils.preprocessing import (
    cap_class_samples,
    detect_label_column,
    ensure_output_dir,
    keep_numeric_features,
    normalize_label,
    remove_correlated_features,
    stratified_split,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(cfg: dict, logger):
    roots = [cfg["dataset"]["folder_1"], cfg["dataset"]["folder_2"]]
    pattern = cfg["dataset"].get("file_pattern", "*.csv")
    frames = []
    for root in roots:
        root_path = Path(root)
        files = sorted(root_path.rglob(pattern))
        logger.info("Scanning %s (%d CSV files)", root, len(files))
        for file in files:
            try:
                frames.append(pd.read_csv(file, low_memory=False))
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Skipping %s due to read error: %s", file, exc)

    if not frames:
        raise FileNotFoundError("No readable CSV files found in configured dataset folders")

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d rows x %d columns", combined.shape[0], combined.shape[1])
    return combined


def preprocess_data(df: pd.DataFrame, cfg: dict, logger) -> pd.DataFrame:
    prep_cfg = cfg["preprocessing"]
    labels_cfg = cfg["labels"]

    label_col = detect_label_column(df, prep_cfg["label_column_candidates"])
    df[label_col] = df[label_col].map(normalize_label)

    allowed = set(labels_cfg["allowed_classes"])
    mapping = labels_cfg.get("mapping", {})
    df[label_col] = df[label_col].map(lambda x: mapping.get(x, x))
    df = df[df[label_col].isin(allowed)].copy()
    if df.empty:
        raise ValueError("No rows remaining after label filtering")

    logger.info("Class distribution before cap: %s", df[label_col].value_counts().to_dict())

    df = cap_class_samples(
        df,
        label_col=label_col,
        max_samples=int(prep_cfg["max_samples_per_class"]),
        random_state=int(prep_cfg["random_state"]),
    )

    logger.info("Class distribution after cap: %s", df[label_col].value_counts().to_dict())

    df = keep_numeric_features(df, label_col=label_col, drop_cols=prep_cfg.get("drop_columns", []))
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna(axis=1, how="all").dropna()

    reduced, dropped = remove_correlated_features(
        df,
        threshold=float(prep_cfg["correlation_threshold"]),
        ignore_cols=[label_col],
    )
    logger.info("Dropped %d highly correlated features", len(dropped))
    return reduced.rename(columns={label_col: "target"})


def save_splits(df: pd.DataFrame, cfg: dict, logger) -> None:
    prep_cfg = cfg["preprocessing"]
    out_dir = ensure_output_dir(cfg["dataset"]["output_dir"])

    train_df, val_df, test_df = stratified_split(
        df,
        label_col="target",
        train_ratio=float(prep_cfg["train_ratio"]),
        val_ratio=float(prep_cfg["val_ratio"]),
        test_ratio=float(prep_cfg["test_ratio"]),
        random_state=int(prep_cfg["random_state"]),
    )

    final_path = out_dir / cfg["dataset"]["final_dataset"]
    train_path = out_dir / "train_data.csv"
    val_path = out_dir / "val_data.csv"
    test_path = out_dir / "test_data.csv"

    df.to_csv(final_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Saved final dataset to %s", final_path)
    logger.info("Saved splits: train=%s val=%s test=%s", train_path, val_path, test_path)


def run(config_path: str):
    cfg = load_config(config_path)
    logger = setup_logger("data_pipeline", "logs/data_pipeline.log")
    logger.info("Starting data pipeline")

    raw_df = load_raw_data(cfg, logger)
    processed_df = preprocess_data(raw_df, cfg, logger)
    save_splits(processed_df, cfg, logger)

    logger.info("Data pipeline completed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare multiclass DDoS data splits")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config)
