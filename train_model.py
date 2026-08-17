import pickle
import re
import time
import mlflow
import warnings
from prepare_data import *
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
import mlflow.sklearn
from optuna.samplers import TPESampler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from data_loader import build_modeling_tables
from modeling import get_models, optimize_LightGBM_with_optuna
from preprocessing import (
    FEATURE_ENGINEERING,
    get_train_test_data,
    prepare_full_train_test_for_submission,
)

warnings.filterwarnings("ignore")
def run_training_pipeline(
    final_feature_count: int = 150,
    n_trials: int = 1,
    timeout: int = 4000,
    submission_path: str = "submission.csv",
):
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
    x_train_copy = x_train.copy()

    results_baseline = {}
    cat_cols = train.drop(columns=["TARGET"]).select_dtypes(include=["object"]).columns
    models = get_models()
    for name, info in models.items():
        print(f"Сейчас обучаем {name}")
        with mlflow.start_run(run_name=name):
            model = info["model"]
            mlflow.log_param("model_name",name)
            if name == "CatBoost":
                model.fit(
                    x_train_for_catboost,
                    y_train_for_catboost,
                    cat_features=cat_cols.to_list(),
                )
                y_proba = model.predict_proba(x_test_for_catboost)[:, 1]
                y_pred = model.predict(x_test_for_catboost)
                x_train_for_catboost = x_train
            else:
                if info["scaled"]:
                    scaler = StandardScaler()
                    x_train = scaler.fit_transform(x_train)
                    x_test = scaler.transform(x_test)
                    x_train_for_not_catboost = x_train
                    x_test_for_not_catboost = x_test

                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                y_proba = model.predict_proba(x_test)[:, 1]

            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_proba)
            fpr, tpr, threshold_roc = roc_curve(y_test, y_proba)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric('roc_auc', roc_auc)
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('pr_auc', pr_auc)
            mlflow.sklearn.log_model(model, "model")
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
            with open(f"models/baseline/{name}.pkl", "wb") as f:
                pickle.dump(model, f)

    feature_importance = pd.DataFrame(
        {
            "feature": x_train_copy.columns,
            "importance": models["LightGBM"]["model"].feature_importances_,
        },
    ).sort_values("importance", ascending=False)

    x_train_for_not_catboost_df = pd.DataFrame(
        x_train_for_not_catboost,
        columns=x_train_copy.columns,
    )
    x_test_for_not_catboost_df = pd.DataFrame(
        x_test_for_not_catboost,
        columns=x_train_copy.columns,
    )

    clean_columns = [
        re.sub(r"[^A-Za-z0-9_]+", "_", col)
        for col in x_train_for_not_catboost_df.columns
    ]
    x_train_for_not_catboost_df.columns = clean_columns
    x_test_for_not_catboost_df.columns = clean_columns
    feature_importance["feature"] = [
        re.sub(r"[^A-Za-z0-9_]+", "_", col)
        for col in feature_importance["feature"]
    ]

    results = []
    feature_counts = [1, 2, 10, 15, 20, 30, 50, 80, 120, 150, 200, 250, 300, 350, 382]

    for n in feature_counts:
        top_n_features = feature_importance.head(n)["feature"].tolist()

        x_train_top = x_train_for_not_catboost_df[top_n_features]
        x_test_top = x_test_for_not_catboost_df[top_n_features]

        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

        model.fit(x_train_top, y_train)
        y_proba = model.predict_proba(x_test_top)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        results.append(
            {
                "n_features": n,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            },
        )

    results_df = pd.DataFrame(results)
    final_features = feature_importance.head(final_feature_count)["feature"].tolist()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    x_train_for_not_catboost_df_opt = x_train_for_not_catboost_df[final_features]
    optimized_results = {}
    optimization_studies = {}
    optimization_times = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    optimization_functions = {
        "LightGBM": optimize_LightGBM_with_optuna,
    }

    start_total_time = time.time()

    for model_name, func in optimization_functions.items():
        print(f"Начинаем Оптимизацию {model_name}")
        start_time = time.time()

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
        )
        study.optimize(
            lambda trial: func(trial, x_train_for_not_catboost_df_opt, y_train),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )

        end_time = time.time()
        optimization_time = end_time - start_time
        optimization_times[model_name] = optimization_time
        optimization_studies[model_name] = study

        best_params = study.best_params
        best_score = study.best_value

        optimized_model = LGBMClassifier(**best_params)
        optimized_model.fit(x_train_for_not_catboost_df_opt, y_train)

        y_pred_opt = optimized_model.predict(x_test_for_not_catboost_df[final_features])
        y_pred_proba_opt = optimized_model.predict_proba(
            x_test_for_not_catboost_df[final_features],
        )[:, 1]

        precision_scores = cross_val_score(
            optimized_model,
            x_train_for_not_catboost_df_opt,
            y_train,
            cv=cv,
            scoring="average_precision",
        )

        roc_auc_current = roc_auc_score(y_test, y_pred_proba_opt)

        optimized_results[model_name] = {
            "best_params": best_params,
            "best_precision_score": best_score,
            "precision_scores": precision_scores,
            "roc_auc": roc_auc_current,
            "predictions": y_pred_opt,
            "probabilities": y_pred_proba_opt,
            "n_trials": len(study.trials),
            "optimization_time": optimization_time,
            "model": optimized_model,
        }
        with mlflow.start_run(run_name="optimized_lightgbm"):
            mlflow.log_params(best_params)
            mlflow.log_param("n_features", len(final_features))
            mlflow.sklearn.log_model( optimized_results['LightGBM']['model'], "model_after_optuna")
            mlflow.log_metric("best_precision_score", optimized_results["LightGBM"]["best_precision_score"])
            mlflow.log_metric("validation_roc_auc", optimized_results["LightGBM"]["roc_auc"])
    total_time = time.time() - start_total_time

    test_ids = test["SK_ID_CURR"].copy()
    TRAIN, TARGET, TEST = prepare_full_train_test_for_submission(
        train_for_learning_before_test,
        test,
    )

    best_params = optimized_results["LightGBM"]["best_params"]
    model_for_test = LGBMClassifier(**best_params)
    model_for_test.fit(TRAIN[final_features], TARGET)
    with open("models/lightgbm_model.pkl", "wb") as f:
        pickle.dump(model_for_test,f)
    with mlflow.start_run(run_name="final_lightgbm"):
        mlflow.sklearn.log_model(model_for_test,"model_for_kaggle")
    test_proba = model_for_test.predict_proba(TEST[final_features])[:, 1]

    submission = pd.DataFrame(
        {
            "SK_ID_CURR": test_ids,
            "TARGET": test_proba,
        },
    )
    submission.to_csv(submission_path, index=False)

    return {
        "train": train,
        "test": test,
        "train_for_learning_before_test": train_for_learning_before_test,
        "baseline_results": results_baseline,
        "results_baseline": results_baseline,
        "feature_importance": feature_importance,
        "results_df": results_df,
        "feature_count_results": results_df,
        "final_features": final_features,
        "optimized_results": optimized_results,
        "optuna_results": optimized_results["LightGBM"],
        "optimization_studies": optimization_studies,
        "optimization_times": optimization_times,
        "total_time": total_time,
        "submission": submission,
    }
if __name__ == "__main__":
    run_training_pipeline()
