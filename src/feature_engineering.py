from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.logging_utils import setup_logger

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=["target"], errors="ignore")
    numeric = features.select_dtypes(include=[np.number])

    out = df.copy()
    out["row_mean"] = numeric.mean(axis=1)
    out["row_std"] = numeric.std(axis=1)
    out["row_skew"] = numeric.skew(axis=1)
    out["row_kurtosis"] = numeric.kurt(axis=1)
    out["row_entropy"] = numeric.apply(lambda r: entropy(np.abs(r) + 1e-12), axis=1)
    return out


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Flow Duration" in out.columns and "Total Fwd Packets" in out.columns:
        out["packet_rate"] = out["Total Fwd Packets"] / (out["Flow Duration"].replace(0, np.nan))
    if "Flow IAT Mean" in out.columns and "Flow IAT Std" in out.columns:
        out["iat_cv"] = out["Flow IAT Std"] / (out["Flow IAT Mean"].replace(0, np.nan))
    return out.fillna(0)


def add_protocol_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Protocol" in out.columns and "Destination Port" in out.columns:
        out["proto_port_interaction"] = out["Protocol"] * out["Destination Port"]
    if "Flow Packets/s" in out.columns and "Flow Bytes/s" in out.columns:
        out["bytes_per_packet"] = out["Flow Bytes/s"] / (out["Flow Packets/s"].replace(0, np.nan))
    return out.fillna(0)


def add_interaction_features(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    out = df.copy()
    candidates = [c for c in out.select_dtypes(include=[np.number]).columns if c != "target"]
    for i in range(min(len(candidates), top_n - 1)):
        c1 = candidates[i]
        c2 = candidates[i + 1]
        out[f"{c1}_x_{c2}"] = out[c1] * out[c2]
    return out


def shap_or_mi_feature_selection(df: pd.DataFrame, logger, max_features: int = 120) -> pd.DataFrame:
    if "target" not in df.columns:
        return df

    x = df.drop(columns=["target"]).select_dtypes(include=[np.number]).fillna(0)
    y = df["target"]
    if x.shape[1] <= max_features:
        return df

    if shap is not None:
        try:
            model = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
            model.fit(x, y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x.sample(min(5000, len(x)), random_state=42))
            if isinstance(shap_values, list):
                importances = np.mean([np.abs(v).mean(axis=0) for v in shap_values], axis=0)
            else:
                importances = np.abs(shap_values).mean(axis=0)
            selected = x.columns[np.argsort(importances)[-max_features:]].tolist()
            logger.info("Selected top %d features with SHAP", len(selected))
            return df[selected + ["target"]]
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("SHAP selection failed, falling back to MI: %s", exc)

    mi = mutual_info_classif(x, y, random_state=42)
    selected = x.columns[np.argsort(mi)[-max_features:]].tolist()
    logger.info("Selected top %d features with mutual information", len(selected))
    return df[selected + ["target"]]


def run(config_path: str):
    logger = setup_logger("feature_engineering", "logs/feature_engineering.log")
    cfg = load_config(config_path)
    base = Path(cfg["dataset"]["output_dir"]) / cfg["dataset"]["final_dataset"]

    if not base.exists():
        raise FileNotFoundError(f"Input dataset not found: {base}. Run data_pipeline first.")

    df = pd.read_csv(base)
    logger.info("Loaded %s with shape %s", base, df.shape)

    df = add_statistical_features(df)
    df = add_temporal_features(df)
    df = add_protocol_features(df)
    df = add_interaction_features(df)
    df = shap_or_mi_feature_selection(df, logger)

    out_path = base.with_name("final_multiclass_data_engineered.csv")
    df.to_csv(out_path, index=False)
    logger.info("Saved engineered dataset to %s (%s)", out_path, df.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate advanced multiclass features")
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config)
