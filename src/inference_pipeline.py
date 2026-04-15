from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Iterable

import joblib
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.logging_utils import setup_logger


@dataclass
class PredictionResult:
    label: str
    confidence: float
    passed_threshold: bool


class InferencePipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.logger = setup_logger("inference_pipeline", "logs/inference_pipeline.log")
        inf_cfg = self.cfg["inference"]

        self.threshold = float(inf_cfg.get("default_threshold", 0.55))
        self.stage1_model = self._safe_load(inf_cfg.get("stage1_model_path"))
        self.multiclass_model = self._safe_load(inf_cfg["multiclass_model_path"])
        self.label_encoder = self._safe_load(Path(self.cfg["models"]["output_dir"]) / "label_encoder.joblib")
        if self.multiclass_model is None or self.label_encoder is None:
            raise FileNotFoundError("Required multiclass model artifacts are missing. Run training first.")

        self.alert_path = Path(inf_cfg["alerts_output"])
        self.alert_path.parent.mkdir(parents=True, exist_ok=True)

        self.registry_path = Path(inf_cfg["model_registry_path"])
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._update_registry()

    def _safe_load(self, path: str | Path | None):
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            self.logger.warning("Model not found: %s", p)
            return None
        return joblib.load(p)

    def _update_registry(self):
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "multiclass_model": str(self.cfg["inference"]["multiclass_model_path"]),
            "stage1_model": str(self.cfg["inference"].get("stage1_model_path")),
            "threshold": self.threshold,
        }
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _binary_stage1_pass(self, features: pd.DataFrame) -> bool:
        if self.stage1_model is None:
            return True
        pred = self.stage1_model.predict(features)
        return bool(pred[0] == 1)

    def predict(self, sample: pd.DataFrame, threshold: float | None = None) -> PredictionResult:
        thr = self.threshold if threshold is None else threshold

        if not self._binary_stage1_pass(sample):
            return PredictionResult(label="benign", confidence=1.0, passed_threshold=True)

        proba = self.multiclass_model.predict_proba(sample)[0]
        idx = int(proba.argmax())
        conf = float(proba[idx])
        label = str(self.label_encoder.inverse_transform([idx])[0])
        passed = conf >= thr

        if passed:
            self._emit_alert(label, conf)

        return PredictionResult(label=label, confidence=conf, passed_threshold=passed)

    def predict_batch(self, df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        thr = self.threshold if threshold is None else threshold
        if len(df) == 0:
            return pd.DataFrame(columns=["prediction", "confidence", "alert"])

        if self.stage1_model is not None:
            stage1_pred = self.stage1_model.predict(df)
            stage1_mask = pd.Series(stage1_pred == 1, index=df.index)
        else:
            stage1_mask = pd.Series(True, index=df.index)

        proba = self.multiclass_model.predict_proba(df)
        idx = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        labels = self.label_encoder.inverse_transform(idx)

        records = []
        for i, row_id in enumerate(df.index):
            if not bool(stage1_mask[row_id]):
                records.append({"prediction": "benign", "confidence": 1.0, "alert": True})
                continue
            passed = bool(conf[i] >= thr)
            if passed:
                self._emit_alert(str(labels[i]), float(conf[i]))
            records.append({"prediction": str(labels[i]), "confidence": float(conf[i]), "alert": passed})
        return pd.DataFrame(records, index=df.index)

    def predict_stream(self, batches: Iterable[pd.DataFrame], threshold: float | None = None):
        for batch in batches:
            yield self.predict_batch(batch, threshold=threshold)

    def _emit_alert(self, label: str, confidence: float):
        msg = f"{datetime.now(timezone.utc).isoformat()} | ALERT | {label} | confidence={confidence:.4f}\n"
        with open(self.alert_path, "a", encoding="utf-8") as f:
            f.write(msg)
        self.logger.warning(msg.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiclass inference")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input", required=True, help="CSV file to score")
    parser.add_argument("--output", default="reports/inference_predictions.csv")
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = InferencePipeline(config_path=args.config)
    data = pd.read_csv(args.input)
    out = pipeline.predict_batch(data, threshold=args.threshold)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    pipeline.logger.info("Saved predictions to %s", out_path)


if __name__ == "__main__":
    main()
