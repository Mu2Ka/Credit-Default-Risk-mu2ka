from pathlib import Path

import pandas as pd

from features import (
    merge_POS_CASH_BALANCE_features,
    merge_bureau_balance_features,
    merge_bureau_features,
    merge_credit_credit_features,
    merge_installments_payments_features,
    merge_previous_application_features,
)


def load_application_train(data_dir: str | Path = ".") -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "application_train.csv")


def load_application_test(data_dir: str | Path = ".") -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "application_test.csv")


def load_bureau(data_dir: str | Path = ".") -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "bureau.csv")


def load_bureau_balance(data_dir: str | Path = ".") -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "bureau_balance.csv")


def load_credit_card_balance(data_dir: str | Path = ".") -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "credit_card_balance.csv")


def load_installments_payments(data_dir: str | Path = ".") -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "installments_payments.csv")


def load_POS_CASH_balance(data_dir: str | Path = ".") -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "POS_CASH_balance.csv")


def load_previous_application(data_dir: str | Path = ".") -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "previous_application.csv")


def load_core_tables(data_dir: str | Path = ".") -> dict[str, pd.DataFrame]:
    return {
        "application_test": load_application_test(data_dir),
        "test": load_application_test(data_dir),
        "application_train": load_application_train(data_dir),
        "bureau": load_bureau(data_dir),
        "bureau_balance": load_bureau_balance(data_dir),
        "credit_card_balance": load_credit_card_balance(data_dir),
        "installments_payments": load_installments_payments(data_dir),
        "POS_CASH_balance": load_POS_CASH_balance(data_dir),
        "previous_application": load_previous_application(data_dir),
    }


def build_modeling_tables(
    data_dir: str | Path = ".",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = load_core_tables(data_dir)

    train = merge_bureau_features(tables["application_train"], tables["bureau"])
    test = merge_bureau_features(tables["application_test"], tables["bureau"])

    train = merge_bureau_balance_features(
        train,
        tables["bureau"],
        tables["bureau_balance"],
    )
    test = merge_bureau_balance_features(
        test,
        tables["bureau"],
        tables["bureau_balance"],
    )

    train = merge_credit_credit_features(train, tables["credit_card_balance"])
    test = merge_credit_credit_features(test, tables["credit_card_balance"])

    train = merge_installments_payments_features(
        train,
        tables["installments_payments"],
    )
    test = merge_installments_payments_features(
        test,
        tables["installments_payments"],
    )

    train = merge_POS_CASH_BALANCE_features(train, tables["POS_CASH_balance"])
    test = merge_POS_CASH_BALANCE_features(test, tables["POS_CASH_balance"])

    train = merge_previous_application_features(
        train,
        tables["previous_application"],
    )
    test = merge_previous_application_features(
        test,
        tables["previous_application"],
    )

    return train, test
