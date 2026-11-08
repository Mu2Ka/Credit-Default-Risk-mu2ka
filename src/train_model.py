import json
import pickle
from pathlib import Path
import re
import time
import mlflow
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import prepare_data
from modeling import get_models
from preprocessing import get_train_test_data, prepare_features
import dagshub
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
print('Старт модуля train_model')
def train_model():
    dagshub.init(repo_owner='Mu2Ka', repo_name='Credit-Default-Risk-mu2ka', mlflow=True)
    try:
        train = pd.read_csv("data/processed/train_processed.csv")
        test = pd.read_csv("data/processed/test_processed.csv")
    except Exception as e:
        logger.warning(f"Processed data not found. Preparing data: {e}")
        logger.info("Preparing data...")
        prepare_data()
        train = pd.read_csv("data/processed/train_processed.csv")
        test = pd.read_csv("data/processed/test_processed.csv")
    train_for_learning_before_test = train.copy()
    (
        x_train,
        x_test,
        y_train,
        y_test,
        x_train_for_catboost,
        x_test_for_catboost,
        y_train_for_catboost,
        y_test_for_catboost,
    ) = get_train_test_data(train)
    print('Получили датасеты')
    x_train_copy = x_train.copy()
    results_baseline = {}
    cat_cols = train.drop(columns=["TARGET"]).select_dtypes(include=["object"]).columns
    models = get_models()
    print('Начинаем обучение моделей')
    for name, info in models.items():
        print(f"Now we start training {name}")
        with mlflow.start_run(run_name=f'baseline_{name}'):
            model = info["model"]

            mlflow.set_tag("model_name",name)
            mlflow.set_tag("stage","baseline")
            if name == "CatBoost":
                model.fit(
                    x_train_for_catboost,
                    y_train_for_catboost,
                    cat_features=cat_cols.to_list(),
                )
                y_proba = model.predict_proba(x_test_for_catboost)[:, 1]
                y_pred = model.predict(x_test_for_catboost)
            else:
                if info["scaled"]:
                    model = make_pipeline(StandardScaler(), model)

                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                y_proba = model.predict_proba(x_test)[:, 1]

            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_proba)
            fpr, tpr, threshold_roc = roc_curve(y_test, y_proba)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model: {name}, Accuracy: {accuracy}, ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
            
            mlflow.log_metric('roc_auc', roc_auc)
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('pr_auc', pr_auc)
            results_baseline[name] = {
                "model": model,
                "accuracy": accuracy,
                "precisions": precisions,
                "recalls": recalls,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "fpr": fpr,
                "tpr": tpr,
                "threshold_roc": threshold_roc,
                "thresholds_pr": thresholds_pr,
            }
            path = Path(f'models/baseline/{name.lower().replace(" ", "_")}_baseline_model.pkl')
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path,'wb') as f:
                pickle.dump(model,f)
            mlflow.sklearn.log_model(
                model,
                name=f"{name.lower().replace(' ', '_')}_baseline_model",
                registered_model_name="credit_default_risk_baseline_model",
                serialization_format="cloudpickle",
            )


def compare_models(results_baseline, x_train, y_train, path="cv_results.json", cv=5):
    scores = {name: [] for name in results_baseline}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(x_train, y_train), start=1):
        x_fold_train, x_fold_valid, x_cat_train, x_cat_valid = prepare_features(
            x_train.iloc[train_idx], x_train.iloc[valid_idx]
        )
        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[valid_idx]
        cat_cols = x_cat_train.select_dtypes(include=["object"]).columns.tolist()

        for name, info in results_baseline.items():
            model = clone(info["model"])
            if name == "CatBoost":
                model.fit(x_cat_train, y_fold_train, cat_features=cat_cols)
                probabilities = model.predict_proba(x_cat_valid)[:, 1]
            else:
                if info.get("scaled"):
                    model = make_pipeline(StandardScaler(), model)
                model.fit(x_fold_train, y_fold_train)
                probabilities = model.predict_proba(x_fold_valid)[:, 1]

            score = roc_auc_score(y_fold_valid, probabilities)
            scores[name].append(float(score))
            print(f"Fold {fold}, {name}: ROC AUC = {score:.4f}")

    results = {}
    for name, fold_scores in scores.items():
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        results[name] = {
            "roc_auc_mean": float(mean_score),
            "roc_auc_std": float(std_score),
            "fold_scores": fold_scores,
        }
        print(f"Model: {name}, Cross-validated ROC AUC: {mean_score:.4f} ± {std_score:.4f}")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    return results


if __name__ == "__main__":
    train = pd.read_csv("data/processed/train_processed.csv")
    X = train.drop(columns=["TARGET"])
    y = train["TARGET"]
    x_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    compare_models(get_models(), x_train, y_train)
