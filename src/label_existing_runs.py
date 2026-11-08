import dagshub
from mlflow.tracking import MlflowClient


runs = {
    "ed4d9b9ebcd3427b9cadaf4ee65eb089": ("LogisticRegression", "baseline"),
    "fe495fb84d9d497bb911f86e5a846f84": ("RandomForest", "baseline"),
    "9bea9c4a1f2b4c0abc90585d7ab99d08": ("CatBoost", "baseline"),
    "3a579593036a43178ce7337cbdc7f7d6": ("LightGBM", "baseline"),
    "98429690ff014b158f2ab9a35e711755": ("LightGBM", "baseline"),
    "c4b7f173c050489495cca239fa625be6": ("LightGBM", "credit_to_annuity"),
    "a0199410be9d41b7b850d762ae6351b9": ("LightGBM", "annuity_debt"),
    "49b91c31cf2b4a2ca9104438c54fb344": ("LightGBM", "annuity_debt_app_3"),
    "774b1130a31b4a279383980c2f80d5f9": ("CatBoost", "annuity_debt_app_3"),
    "14873a10cfe74027b0df7c1a6c697058": ("LightGBM", "annuity_debt_app_3"),
    "adff0421ef714c64a10a551833f5ad65": ("CatBoost", "annuity_debt_app_3"),
}


if __name__ == "__main__":
    dagshub.init(repo_owner="Mu2Ka", repo_name="Credit-Default-Risk-mu2ka", mlflow=True)
    client = MlflowClient()
    for run in client.search_runs(experiment_ids=["0"], max_results=1000):
        run_id = run.info.run_id
        if run_id in runs:
            model, feature_set = runs[run_id]
            client.set_tag(run_id, "model_name", model)
            client.set_tag(run_id, "feature_set", feature_set)
            client.set_tag(run_id, "evaluation", "holdout")
            client.set_tag(run_id, "split_seed", "42")
            client.set_tag(run_id, "test_size", "0.2")
            client.set_tag(run_id, "stage", "baseline" if feature_set == "baseline" else "experiment")
            print(f"Updated {model}: {feature_set}")
        elif "roc_auc" in run.data.metrics and "evaluation" not in run.data.tags:
            client.set_tag(run_id, "evaluation", "holdout")
            print(f"Marked holdout run: {run_id}")
