from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.logging_utils import setup_logger
from src.utils.metrics import focal_like_sample_weights

try:
    from imblearn.over_sampling import SMOTE  # type: ignore
except Exception:  # pragma: no cover
    SMOTE = None

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None

try:
    from catboost import CatBoostClassifier  # type: ignore
except Exception:  # pragma: no cover
    CatBoostClassifier = None

try:
    import optuna  # type: ignore
except Exception:  # pragma: no cover
    optuna = None


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_splits(cfg: dict):
    data_dir = Path(cfg["dataset"]["output_dir"])
    train = pd.read_csv(data_dir / "train_data.csv")
    val = pd.read_csv(data_dir / "val_data.csv")
    test = pd.read_csv(data_dir / "test_data.csv")
    return train, val, test


def prepare_xy(df: pd.DataFrame, encoder: LabelEncoder | None = None):
    x = df.drop(columns=["target"])
    y_raw = df["target"].astype(str)
    if encoder is None:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw)
    else:
        y = encoder.transform(y_raw)
    return x, y, encoder


def build_base_models(cfg: dict, class_weights: dict[int, float], num_classes: int):
    mcfg = cfg["models"]
    models = {}

    rf_cfg = mcfg["random_forest"]
    models["rf"] = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_split=rf_cfg["min_samples_split"],
        n_jobs=rf_cfg.get("n_jobs", -1),
        class_weight=class_weights,
        random_state=mcfg["random_state"],
    )

    if xgb is not None:
        xcfg = mcfg["xgboost"]
        models["xgb"] = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            n_estimators=xcfg["n_estimators"],
            max_depth=xcfg["max_depth"],
            learning_rate=xcfg["learning_rate"],
            subsample=xcfg["subsample"],
            colsample_bytree=xcfg["colsample_bytree"],
            tree_method=xcfg.get("tree_method", "hist"),
            random_state=mcfg["random_state"],
        )

    if lgb is not None:
        lcfg = mcfg["lightgbm"]
        models["lightgbm"] = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=lcfg["n_estimators"],
            learning_rate=lcfg["learning_rate"],
            num_leaves=lcfg["num_leaves"],
            subsample=lcfg["subsample"],
            colsample_bytree=lcfg["colsample_bytree"],
            class_weight={int(k): float(v) for k, v in class_weights.items()},
            random_state=mcfg["random_state"],
        )

    if CatBoostClassifier is not None:
        ccfg = mcfg["catboost"]
        models["catboost"] = CatBoostClassifier(
            iterations=ccfg["iterations"],
            learning_rate=ccfg["learning_rate"],
            depth=ccfg["depth"],
            loss_function=ccfg["loss_function"],
            verbose=False,
            random_seed=mcfg["random_state"],
        )

    return models


def maybe_apply_smote(x, y, cfg, logger):
    if not cfg["models"].get("use_smote", True) or SMOTE is None:
        return x, y
    smote = SMOTE(random_state=cfg["models"]["random_state"])
    x_res, y_res = smote.fit_resample(x, y)
    logger.info("Applied SMOTE: %s -> %s", x.shape, x_res.shape)
    return x_res, y_res


def optimize_rf_with_optuna(x, y, cfg, logger):
    if optuna is None:
        return None

    def objective(trial):
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 700),
            max_depth=trial.suggest_int("max_depth", 6, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            n_jobs=-1,
            random_state=cfg["models"]["random_state"],
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg["models"]["random_state"])
        return cross_val_score(model, x, y, cv=cv, scoring="f1_macro", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(cfg["models"].get("optuna_trials", 100)))
    logger.info("Optuna RF best score: %.4f", study.best_value)
    return study.best_params


def train_and_score(models, x_train, y_train, x_val, y_val, cfg, logger):
    results = {}
    gamma = float(cfg["models"].get("focal_gamma", 2.0))
    sample_weights = focal_like_sample_weights(y_train, gamma=gamma)

    for name, model in models.items():
        try:
            if name in {"xgb", "lightgbm", "catboost", "rf"}:
                model.fit(x_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(x_train, y_train)
        except TypeError:
            model.fit(x_train, y_train)

        pred = model.predict(x_val)
        score = f1_score(y_val, pred, average="macro")
        results[name] = {"model": model, "val_macro_f1": float(score)}
        logger.info("%s validation macro-F1: %.4f", name, score)

    return results


def build_stacking(results, cfg, random_state=42):
    estimators = [(k, v["model"]) for k, v in results.items()]
    if not estimators:
        raise ValueError("No trained base estimators available for stacking")

    final_estimator = LogisticRegression(max_iter=300, multi_class="multinomial")
    if cfg["models"]["ensemble"].get("meta_learner") == "xgboost" and xgb is not None:
        final_estimator = xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=250,
            learning_rate=0.07,
            max_depth=5,
            random_state=random_state,
        )

    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=3,
        n_jobs=-1,
        stack_method="predict_proba",
    )


def train_ovr_for_weak_classes(x_train, y_train, encoder: LabelEncoder, weak_classes: list[str], logger):
    weak_indices = [encoder.transform([c])[0] for c in weak_classes if c in encoder.classes_]
    if not weak_indices:
        return None

    base = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=300, class_weight="balanced")),
        ]
    )
    ovr = OneVsRestClassifier(base)
    ovr.fit(x_train, y_train)
    logger.info("Trained One-vs-Rest helper for weak classes: %s", weak_classes)
    return {"model": ovr, "weak_class_indices": weak_indices}


def run(config_path: str):
    cfg = load_config(config_path)
    logger = setup_logger("model_training", "logs/model_training.log")
    model_dir = Path(cfg["models"]["output_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_splits(cfg)
    x_train, y_train, encoder = prepare_xy(train_df)
    x_val, y_val, _ = prepare_xy(val_df, encoder)
    x_test, y_test, _ = prepare_xy(test_df, encoder)

    classes = np.unique(y_train)
    class_weight_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, class_weight_vals)}

    x_train_bal, y_train_bal = maybe_apply_smote(x_train, y_train, cfg, logger)

    best_rf_params = optimize_rf_with_optuna(x_train_bal, y_train_bal, cfg, logger)
    models = build_base_models(cfg, class_weights, num_classes=len(encoder.classes_))
    if best_rf_params:
        models["rf"].set_params(**best_rf_params)

    results = train_and_score(models, x_train_bal, y_train_bal, x_val, y_val, cfg, logger)

    stacker = build_stacking(results, cfg, random_state=cfg["models"]["random_state"])
    stacker.fit(x_train_bal, y_train_bal)
    stack_pred = stacker.predict(x_val)
    stack_score = f1_score(y_val, stack_pred, average="macro")
    logger.info("stacking validation macro-F1: %.4f", stack_score)

    weak_ovr = train_ovr_for_weak_classes(x_train_bal, y_train_bal, encoder, cfg["models"]["weak_classes"], logger)

    final_pred = stacker.predict(x_test)
    final_macro_f1 = f1_score(y_test, final_pred, average="macro")
    logger.info("test macro-F1: %.4f", final_macro_f1)

    artifacts = {
        "label_encoder": encoder,
        "base_models": {k: v["model"] for k, v in results.items()},
        "stacking_model": stacker,
        "weak_ovr": weak_ovr,
    }

    for name, model in artifacts.items():
        joblib.dump(model, model_dir / f"{name}.joblib")

    metrics = {
        "val_scores": {k: v["val_macro_f1"] for k, v in results.items()},
        "stacking_val_macro_f1": float(stack_score),
        "test_macro_f1": float(final_macro_f1),
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved model artifacts and metrics to %s", model_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multiclass DDoS ensemble")
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config)
