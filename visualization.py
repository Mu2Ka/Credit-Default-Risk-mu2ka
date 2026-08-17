from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_target_distribution(train: pd.DataFrame) -> None:
    target_counts = train["TARGET"].value_counts()

    plt.figure(figsize=(6, 4))
    plt.bar(target_counts.index.astype(str), target_counts.values)
    plt.title("Distribution of TARGET")
    plt.xlabel("TARGET")
    plt.ylabel("Count")
    plt.grid()
    plt.show()


def plot_categorical_target_rates(
    train: pd.DataFrame,
    columns: list[str] | None = None,
) -> None:
    columns = columns or [
        "CODE_GENDER",
        "NAME_CONTRACT_TYPE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
    ]
    existing_columns = [column for column in columns if column in train.columns]
    colors = ["r", "b", "g", "y", "m"]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    axes = axes.flatten()
    for index, column in enumerate(existing_columns):
        rate = (train.groupby(column)["TARGET"].mean() * 100).sort_values(ascending=False)
        axes[index].bar(rate.index, rate.values, color=colors[index % len(colors)])
        axes[index].set_title(f"Частота TARGET=1 по {column}")
        axes[index].set_ylabel("TARGET=1, %")
        axes[index].tick_params(axis="x", rotation=45)
        axes[index].grid()

    for axis in axes[len(existing_columns):]:
        fig.delaxes(axis)

    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(
    train: pd.DataFrame,
    columns: list[str] | None = None,
) -> None:
    columns = columns or [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
    ]
    existing_columns = [column for column in columns if column in train.columns]
    colors = ["r", "b", "g", "y", "m"]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    axes = axes.flatten()
    for index, column in enumerate(existing_columns):
        axes[index].hist(
            train[column].dropna(),
            bins=30,
            color=colors[index % len(colors)],
            edgecolor="black",
        )
        axes[index].set_title(f"Распределение {column}")
        axes[index].set_ylabel("Количество")
        axes[index].tick_params(axis="x", rotation=45)
        axes[index].grid()

    for axis in axes[len(existing_columns):]:
        fig.delaxes(axis)

    plt.tight_layout()
    plt.show()


def plot_credit_amount_by_target(train: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 10))
    plt.hist(
        train[train["TARGET"] == 1]["AMT_CREDIT"],
        bins=30,
        alpha=0.5,
        label="TARGET 1",
        density=True,
    )
    plt.hist(
        train[train["TARGET"] == 0]["AMT_CREDIT"],
        bins=30,
        alpha=0.5,
        label="TARGET 0",
        density=True,
    )
    plt.title("AMT_CREDIT по TARGET")
    plt.xlabel("AMT_CREDIT")
    plt.ylabel("Количество")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_default_rate_by_credit_amount(
    train: pd.DataFrame,
    quantiles: int = 5,
) -> None:
    credit_bins = pd.qcut(train["AMT_CREDIT"], q=quantiles, duplicates="drop")
    rate = train.groupby(credit_bins, observed=False)["TARGET"].mean() * 100

    plt.figure(figsize=(10, 10))
    plt.plot(rate.index.astype(str), rate.values, marker="o")
    plt.title("Доля TARGET=1 по AMT_CREDIT")
    plt.xlabel("AMT_CREDIT")
    plt.ylabel("TARGET=1, %")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_income_and_age_distributions(train: pd.DataFrame) -> None:
    plot_frame = train.copy()
    if "DAYS_EMPLOYED" in plot_frame.columns:
        plot_frame["DAYS_EMPLOYED"] = plot_frame["DAYS_EMPLOYED"].replace(365243, np.nan)

    plt.figure(figsize=(6, 4))
    plt.hist(np.log1p(plot_frame["AMT_INCOME_TOTAL"].dropna()), bins=30)
    plt.title("Распределение log(AMT_INCOME_TOTAL)")
    plt.xlabel("log(AMT_INCOME_TOTAL)")
    plt.ylabel("Количество")
    plt.grid(alpha=0.3)
    plt.show()

    plot_frame["age_years"] = -plot_frame["DAYS_BIRTH"] / 365
    plt.figure(figsize=(6, 4))
    plt.hist(plot_frame["age_years"].dropna(), bins=30, color="gold")
    plt.title("Распределение возраста")
    plt.xlabel("Возраст, лет")
    plt.ylabel("Количество")
    plt.grid(alpha=0.3)
    plt.show()


def plot_baseline_metric_bars(results_baseline: dict[str, dict[str, object]]) -> None:
    model_names = list(results_baseline.keys())
    pr_aucs = [results_baseline[name]["pr_auc"] for name in model_names]
    roc_aucs = [results_baseline[name]["roc_auc"] for name in model_names]

    plt.figure(figsize=(8, 6))
    plt.bar(model_names, pr_aucs)
    plt.title("PR-AUC by Model")
    plt.xlabel("Model")
    plt.ylabel("PR-AUC")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.bar(model_names, roc_aucs)
    plt.title("ROC-AUC by Model")
    plt.xlabel("Model")
    plt.ylabel("ROC-AUC")
    plt.grid(True)
    plt.show()


def plot_baseline_curves(results_baseline: dict[str, dict[str, object]]) -> None:
    plt.figure(figsize=(8, 6))
    for name, metrics in results_baseline.items():
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{name}: {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    for name, metrics in results_baseline.items():
        plt.plot(
            metrics["recalls"],
            metrics["precisions"],
            label=f"{name}: {metrics['pr_auc']:.3f}",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 50,
) -> None:
    top_features = feature_importance.head(top_n).sort_values("importance", ascending=False)
    plt.figure(figsize=(15, 15))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title(f"Top {top_n} LightGBM Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def plot_feature_count_experiment(results_df: pd.DataFrame) -> None:
    plot_frame = results_df.copy()
    plot_frame["n_features"] = plot_frame["n_features"].astype(str)

    plt.figure(figsize=(8, 5))
    plt.plot(plot_frame["n_features"], plot_frame["roc_auc"], marker="o", label="ROC-AUC")
    plt.plot(plot_frame["n_features"], plot_frame["pr_auc"], marker="o", label="PR-AUC")
    plt.title("LightGBM Performance vs Number of Features")
    plt.xlabel("Number of Features")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()
