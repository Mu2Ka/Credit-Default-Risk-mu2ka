from typing import Any

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def get_models() -> dict[str, dict[str, Any]]:
    return {
        "LogisticRegression": {
            "model": LogisticRegression(
                random_state=42,
                max_iter=1000,
            ),
            "scaled": True,
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
                verbose=0,
            ),
            "scaled": False,
        },
        "LightGBM": {
            "model": LGBMClassifier(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ),
            "scaled": False,
        },
        "CatBoost": {
            "model": CatBoostClassifier(
                iterations=500,
                learning_rate=0.03,
                depth=6,
                eval_metric="PRAUC",
                random_seed=42,
                verbose=False,
            ),
            "scaled": False,
        },
    }


def optimize_LightGBM_with_optuna(trial, x_train, y_train) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1800),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 96),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "random_state": 42,
        "verbose": -1,
    }

    model = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring="average_precision",
    )
    return scores.mean()
