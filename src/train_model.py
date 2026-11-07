import pickle
from pathlib import Path
import re
import time
import mlflow
import warnings
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import prepare_data
from modeling import get_models
from preprocessing import get_train_test_data
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


if __name__ == "__main__":
    train_model()
