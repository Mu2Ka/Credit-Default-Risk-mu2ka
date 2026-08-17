import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def FEATURE_ENGINEERING(dataset: pd.DataFrame) -> pd.DataFrame:
    train = dataset.copy()
    few_passes = [
        "DAYS_LAST_PHONE_CHANGE",
        "CNT_FAM_MEMBERS",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    ]
    train[few_passes] = train[few_passes].fillna(train[few_passes].mean())

    social_columns = [
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
    ]
    train[social_columns] = train[social_columns].fillna(0)

    amt_req = [col for col in train.columns if col.startswith("AMT_REQ_CREDIT_BUREAU")]
    for i in range(len(amt_req)):
        train[f"have_checked_req_credit_{i}"] = (train[amt_req[i]] > 0).astype(int)
    train[amt_req] = train[amt_req].fillna(0)

    ext_sour = [col for col in train.columns if col.startswith("EXT_SOURCE_")]
    for ext in ext_sour:
        train[ext] = train[ext].fillna(train[ext].mean())

    past_end = [col for col in train.columns if col.startswith("PAST_END")]
    early_close = [col for col in train.columns if col.startswith("EARLY_CLOSE")]
    ins_col = [col for col in train.columns if col.startswith("INS")]
    pos_col = [col for col in train.columns if col.startswith("POS")]
    approv_col = [col for col in train.columns if col.startswith("APPROVED")]
    prev_col = [col for col in train.columns if col.startswith("PREV")]

    train[pos_col] = train[pos_col].fillna(0)
    train[approv_col] = train[approv_col].fillna(0)
    train[prev_col] = train[prev_col].fillna(0)
    train[ins_col] = train[ins_col].fillna(0)

    credit_cols = [
        "CREDIT_COUNT",
        "CREDIT_MEAN",
        "CREDIT_MAX",
        "ACTIVE_CREDIT_SUM",
        "ACTIVE_CREDIT_MEAN",
        "CREDIT_DAY_OVERDUE_SUM",
        "CREDIT_DAY_OVERDUE_MAX",
        "CREDIT_DAY_OVERDUE_MIN",
        "CREDIT_DAY_OVERDUE_MEAN",
        "DAYS_CREDIT_ENDDATE_MIN",
        "DAYS_CREDIT_ENDDATE_MAX",
        "DAYS_CREDIT_ENDDATE_MEAN",
        "DAYS_ENDDATE_FACT_MEAN",
    ]
    bureau_features = [
        "CREDIT_SUM",
        "DAYS_CREDIT_MOST_RECENT",
        "DAYS_CREDIT_MEAN",
        "DAYS_CREDIT_OLDEST",
        "DAYS_CREDIT_ENDDATE_SUM",
    ]
    train["OWN_CAR_AGE"] = train["OWN_CAR_AGE"].fillna(0)
    train[credit_cols] = train[credit_cols].fillna(0)
    train[bureau_features] = train[bureau_features].fillna(0)

    for past in past_end:
        train[past] = train[past].fillna(0)
    for early in early_close:
        train[early] = train[early].fillna(0)

    categorical = [
        "NAME_TYPE_SUITE",
        "OCCUPATION_TYPE",
        "EMERGENCYSTATE_MODE",
        "HOUSETYPE_MODE",
        "WALLSMATERIAL_MODE",
        "FONDKAPREMONT_MODE",
    ]
    loan_types = [
        "Another type of loan",
        "Car loan",
        "Loan for business development",
        "Consumer credit",
        "Credit card",
        "Interbank credit",
        "Loan for the purchase of equipment",
        "Loan for purchase of shares (margin lending)",
        "Loan for working capital replenishment",
        "Cash loan (non-earmarked)",
        "Real estate loan",
        "Mortgage",
        "Mobile operator loan",
        "Unknown type of loan",
        "Microloan",
    ]

    train["HAS_POS_HISTORY"] = train["POS_PREV_CREDIT_COUNT"].notna().astype(int)
    train["HAS_PREV_HISTORY"] = train["PREV_APPLICATION_COUNT"].notna().astype(int)
    train["HAS_APPROVED_HISTORY"] = train["APPROVED_AMT_CREDIT_MEAN"].notna().astype(int)

    housing = [
        "TOTALAREA_MODE",
        "YEARS_BEGINEXPLUATATION_AVG",
        "YEARS_BEGINEXPLUATATION_MEDI",
        "YEARS_BEGINEXPLUATATION_MODE",
        "YEARS_BUILD_AVG",
        "YEARS_BUILD_MEDI",
        "YEARS_BUILD_MODE",
        "FLOORSMAX_AVG",
        "FLOORSMAX_MEDI",
        "FLOORSMAX_MODE",
        "FLOORSMIN_AVG",
        "FLOORSMIN_MEDI",
        "FLOORSMIN_MODE",
        "LIVINGAREA_AVG",
        "LIVINGAREA_MEDI",
        "LIVINGAREA_MODE",
        "NONLIVINGAREA_AVG",
        "NONLIVINGAREA_MEDI",
        "NONLIVINGAREA_MODE",
        "APARTMENTS_AVG",
        "APARTMENTS_MEDI",
        "APARTMENTS_MODE",
        "LIVINGAPARTMENTS_AVG",
        "LIVINGAPARTMENTS_MEDI",
        "LIVINGAPARTMENTS_MODE",
        "NONLIVINGAPARTMENTS_AVG",
        "NONLIVINGAPARTMENTS_MEDI",
        "NONLIVINGAPARTMENTS_MODE",
        "BASEMENTAREA_AVG",
        "BASEMENTAREA_MEDI",
        "BASEMENTAREA_MODE",
        "COMMONAREA_AVG",
        "COMMONAREA_MEDI",
        "COMMONAREA_MODE",
        "LANDAREA_AVG",
        "LANDAREA_MEDI",
        "LANDAREA_MODE",
        "ENTRANCES_AVG",
        "ENTRANCES_MEDI",
        "ENTRANCES_MODE",
        "ELEVATORS_AVG",
        "ELEVATORS_MEDI",
        "ELEVATORS_MODE",
    ]

    train[categorical] = train[categorical].fillna("Unknown")
    train[loan_types] = train[loan_types].fillna(0)
    train[housing] = train[housing].fillna(-1)

    bb_cols = [col for col in train.columns if col.startswith("BB_")]
    train[bb_cols] = train[bb_cols].fillna(0)
    train[bb_cols] = train[bb_cols].fillna(0)

    credit_card_cols = [
        "balance_to_limit_mean",
        "balance_to_limit_max",
        "dpd_mean",
        "dpd_max",
        "payment_to_min_mean",
        "payment_to_min_max",
        "payment_to_balance_mean",
        "payment_to_balance_max",
        "drawings_to_limit_mean",
        "drawings_to_limit_max",
    ]
    train[credit_card_cols] = train[credit_card_cols].fillna(0)
    train["DAYS_EMPLOYED"] = train["DAYS_EMPLOYED"].fillna(train["DAYS_EMPLOYED"].median())

    train = train.copy()
    train["credit_income_credit"] = train["AMT_CREDIT"] / train["AMT_INCOME_TOTAL"]
    train["annuity_to_income"] = train["AMT_ANNUITY"] / train["AMT_INCOME_TOTAL"]
    train["income_for_person"] = train["AMT_INCOME_TOTAL"] / train["CNT_FAM_MEMBERS"]
    train["EXT_SOURCE_MEAN"] = train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    train["EXT_SOURCE_MAX"] = train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis=1)
    train["EXT_SOURCE_MIN"] = train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)
    train["EXT_SOURCE_STD"] = train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    train.isnull().sum().sort_values(ascending=False).head(30)
    train = train.replace([np.inf, -np.inf], np.nan).fillna(0)
    return train


def get_train_test_data(train: pd.DataFrame):
    Y = train["TARGET"]
    X = train.drop("TARGET", axis=1)
    cat_cols = X.select_dtypes(include=["object"]).columns

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y,
    )
    x_train_for_catboost = x_train
    x_test_for_catboost = x_test
    y_train_for_catboost = y_train
    y_test_for_catboost = y_test

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_train_ohe = ohe.fit_transform(x_train[cat_cols])
    x_train_ohe_df = pd.DataFrame(
        x_train_ohe,
        columns=ohe.get_feature_names_out(cat_cols),
        index=x_train.index,
    )
    x_test_ohe = ohe.transform(x_test[cat_cols])
    x_test_ohe_df = pd.DataFrame(
        x_test_ohe,
        columns=ohe.get_feature_names_out(cat_cols),
        index=x_test.index,
    )

    x_train = x_train.drop(columns=cat_cols)
    x_test = x_test.drop(columns=cat_cols)

    x_train = pd.concat([x_train, x_train_ohe_df], axis=1)
    x_test = pd.concat([x_test, x_test_ohe_df], axis=1)
    print(x_train.shape, x_test.shape)

    return (
        x_train,
        x_test,
        y_train,
        y_test,
        x_train_for_catboost,
        x_test_for_catboost,
        y_train_for_catboost,
        y_test_for_catboost,
    )


def prepare_full_train_test_for_submission(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    target = train_df["TARGET"].copy()
    train_features = train_df.drop("TARGET", axis=1).copy()
    test_features = test_df.copy()

    cat_cols = train_features.select_dtypes(include=["object"]).columns
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    train_ohe = ohe.fit_transform(train_features[cat_cols])
    train_ohe_df = pd.DataFrame(
        train_ohe,
        columns=ohe.get_feature_names_out(cat_cols),
        index=train_features.index,
    )
    test_ohe = ohe.transform(test_features[cat_cols])
    test_ohe_df = pd.DataFrame(
        test_ohe,
        columns=ohe.get_feature_names_out(cat_cols),
        index=test_features.index,
    )

    train_features = train_features.drop(columns=cat_cols)
    test_features = test_features.drop(columns=cat_cols)

    train_encoded = pd.concat([train_features, train_ohe_df], axis=1)
    test_encoded = pd.concat([test_features, test_ohe_df], axis=1)

    train_encoded.columns = [
        re.sub(r"[^A-Za-z0-9_]+", "_", col)
        for col in train_encoded.columns
    ]
    test_encoded.columns = [
        re.sub(r"[^A-Za-z0-9_]+", "_", col)
        for col in test_encoded.columns
    ]

    train_encoded, test_encoded = train_encoded.align(
        test_encoded,
        join="left",
        axis=1,
        fill_value=0,
    )

    return train_encoded, target, test_encoded


feature_engineering = FEATURE_ENGINEERING
