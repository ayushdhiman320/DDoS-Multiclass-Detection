from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import label_binarize

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


def plot_confusion(cm, labels, out_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run(config_path: str):
    cfg = load_config(config_path)
    logger = setup_logger("model_evaluation", "logs/model_evaluation.log")

    data_dir = Path(cfg["dataset"]["output_dir"])
    model_dir = Path(cfg["models"]["output_dir"])
    report_dir = Path(cfg["models"]["metrics_dir"])
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(data_dir / "test_data.csv")
    train_df = pd.read_csv(data_dir / "train_data.csv")

    encoder = joblib.load(model_dir / "label_encoder.joblib")
    model = joblib.load(model_dir / "stacking_model.joblib")

    x_test = test_df.drop(columns=["target"])
    y_test = encoder.transform(test_df["target"].astype(str))

    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)

    labels = encoder.classes_.tolist()
    cm = confusion_matrix(y_test, pred)
    plot_confusion(cm, labels, fig_dir / "confusion_matrix.png")

    p, r, f1, s = precision_recall_fscore_support(y_test, pred, labels=np.arange(len(labels)), zero_division=0)
    per_class = {
        labels[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(s[i])}
        for i in range(len(labels))
    }

    plt.figure(figsize=(10, 4))
    sns.barplot(x=list(per_class.keys()), y=[v["f1"] for v in per_class.values()], color="steelblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("F1")
    plt.title("Per-class F1")
    plt.tight_layout()
    plt.savefig(fig_dir / "f1_per_class.png")
    plt.close()

    y_bin = label_binarize(y_test, classes=np.arange(len(labels)))
    roc_auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")

    plt.figure(figsize=(9, 7))
    thresholds = {}
    for i, cls in enumerate(labels):
        fpr, tpr, thr = roc_curve(y_bin[:, i], proba[:, i])
        idx = np.argmax(tpr - fpr)
        thresholds[cls] = float(thr[idx])
        plt.plot(fpr, tpr, label=f"{cls} (AUC approx)")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curves.png")
    plt.close()

    # Learning-like CV analysis
    x_train = train_df.drop(columns=["target"])
    y_train = encoder.transform(train_df["target"].astype(str))
    cv = StratifiedKFold(n_splits=int(cfg["models"]["cv_folds"]), shuffle=True, random_state=cfg["models"]["random_state"])
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        x_train,
        y_train,
        cv=3,
        train_sizes=np.linspace(0.2, 1.0, 5),
        scoring="f1_macro",
        n_jobs=-1,
    )
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="train")
    plt.plot(train_sizes, val_scores.mean(axis=1), marker="o", label="validation")
    plt.title("Learning Curve (Macro-F1)")
    plt.xlabel("Training samples")
    plt.ylabel("Macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "learning_curve.png")
    plt.close()

    # Calibration curve (class 0 vs rest)
    CalibrationDisplay.from_predictions((y_test == 0).astype(int), proba[:, 0], n_bins=10)
    plt.tight_layout()
    plt.savefig(fig_dir / "calibration_curve_class0.png")
    plt.close()

    if shap is not None:
        try:
            shap_sample = x_test.sample(min(300, len(x_test)), random_state=42)
            explainer = shap.Explainer(model.predict_proba, shap_sample)
            shap_values = explainer(shap_sample)
            plt.figure()
            shap.summary_plot(shap_values, show=False)
            plt.tight_layout()
            plt.savefig(fig_dir / "shap_summary.png")
            plt.close()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Skipping SHAP visualization: %s", exc)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = np.argsort(importances)[-20:]
        cols = x_test.columns[order]
        vals = importances[order]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=vals, y=cols, orient="h", color="slateblue")
        plt.title("Top Feature Importances")
        plt.tight_layout()
        plt.savefig(fig_dir / "feature_importance.png")
        plt.close()

    output = {
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "macro_roc_auc_ovr": float(roc_auc),
        "classification_report": classification_report(y_test, pred, target_names=labels, zero_division=0),
        "per_class_metrics": per_class,
        "optimized_thresholds": thresholds,
        "cv_macro_f1_scores": [float(s) for s in cv_scores],
        "cv_macro_f1_mean": float(np.mean(cv_scores)),
    }

    (report_dir / "evaluation_report.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Saved evaluation outputs to %s", report_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiclass DDoS models")
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config)
