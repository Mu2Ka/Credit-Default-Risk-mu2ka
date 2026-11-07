import numpy as np
import pandas as pd


def build_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    bureau = bureau.copy()

    bureau["ACTIVE_CREDIT"] = (bureau["DAYS_CREDIT_ENDDATE"] > 0).astype(int)
    bureau["PAST_ENDDATE"] = (bureau["DAYS_CREDIT_ENDDATE"] < 0).astype(int)
    bureau["EARLY_CLOSE"] = (
            bureau["DAYS_ENDDATE_FACT"] > bureau["DAYS_CREDIT_ENDDATE"]
    ).astype(int)

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        CREDIT_COUNT=("SK_ID_BUREAU", "count"),
        CREDIT_MEAN=("AMT_CREDIT_SUM", "mean"),
        CREDIT_MAX=("AMT_CREDIT_SUM", "max"),
        DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        DAYS_CREDIT_MOST_RECENT=("DAYS_CREDIT", "max"),
        DAYS_CREDIT_OLDEST=("DAYS_CREDIT", "min"),
        CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        CREDIT_DAY_OVERDUE_MIN=("CREDIT_DAY_OVERDUE", "min"),
        CREDIT_DAY_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),
        CREDIT_DAY_OVERDUE_SUM=("CREDIT_DAY_OVERDUE", "sum"),
        DAYS_CREDIT_ENDDATE_MAX=("DAYS_CREDIT_ENDDATE", "max"),
        DAYS_CREDIT_ENDDATE_MIN=("DAYS_CREDIT_ENDDATE", "min"),
        DAYS_CREDIT_ENDDATE_MEAN=("DAYS_CREDIT_ENDDATE", "mean"),
        DAYS_CREDIT_ENDDATE_SUM=("DAYS_CREDIT_ENDDATE", "sum"),
        DAYS_ENDDATE_FACT_MEAN=("DAYS_ENDDATE_FACT", "mean"),
        ACTIVE_CREDIT_SUM=("ACTIVE_CREDIT", "sum"),
        ACTIVE_CREDIT_MEAN=("ACTIVE_CREDIT", "mean"),
        PAST_ENDDATE_SUM=("PAST_ENDDATE", "sum"),
        PAST_ENDDATE_MEAN=("PAST_ENDDATE", "mean"),
        EARLY_CLOSE_SUM=("EARLY_CLOSE", "sum"),
        EARLY_CLOSE_MEAN=("EARLY_CLOSE", "mean"),
    ).reset_index()

    type_credit = (
        bureau.groupby(["SK_ID_CURR", "CREDIT_TYPE"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    return bureau_agg.merge(type_credit, on="SK_ID_CURR", how="left")


def merge_bureau_features(
        application_train: pd.DataFrame, bureau: pd.DataFrame
) -> pd.DataFrame:
    bureau_features = build_bureau_features(bureau)
    return application_train.merge(bureau_features, on="SK_ID_CURR", how="left")


def build_bureau_balance_features(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    bureau_balance = bureau_balance.copy()

    bb_agg = bureau_balance.groupby("SK_ID_BUREAU").agg({
        "MONTHS_BALANCE": ["min", "max", "count"]
    })
    bb_agg.columns = ["bb_min", "bb_max", "bb_count"]

    bureau_balance["status_to_num"] = bureau_balance["STATUS"].replace({
        "C": 0,
        "X": 0,
    }).astype(int)
    status_agg = bureau_balance.groupby("SK_ID_BUREAU")["status_to_num"].agg([
        "max",
        "mean",
        "sum",
    ])
    status_agg.columns = ["bb_status_max", "bb_status_mean", "bb_status_sum"]

    bureau_balance["is_bad"] = bureau_balance["STATUS"].isin(
        ["1", "2", "3", "4", "5"]
    ).astype(int)
    bad_agg = bureau_balance.groupby("SK_ID_BUREAU")["is_bad"].mean()
    bad_agg = bad_agg.rename("bb_bad_ratio")

    return (
        bb_agg
        .join(status_agg)
        .join(bad_agg)
    ).reset_index()


def build_bureau_balance_agg_features(
        bureau: pd.DataFrame, bureau_balance: pd.DataFrame
) -> pd.DataFrame:
    bb_features = build_bureau_balance_features(bureau_balance)
    bureau_full = bureau.merge(bb_features, on="SK_ID_BUREAU", how="left")

    return bureau_full.groupby("SK_ID_CURR").agg(
        BB_MIN_MIN=("bb_min", "min"),
        BB_MAX_MAX=("bb_max", "max"),
        BB_COUNT_MEAN=("bb_count", "mean"),
        BB_COUNT_SUM=("bb_count", "sum"),
        BB_STATUS_MAX_MAX=("bb_status_max", "max"),
        BB_STATUS_MEAN_MEAN=("bb_status_mean", "mean"),
        BB_STATUS_SUM_SUM=("bb_status_sum", "sum"),
        BB_BAD_RATIO_MEAN=("bb_bad_ratio", "mean"),
        BB_BAD_RATIO_MAX=("bb_bad_ratio", "max"),
    ).reset_index()


def merge_bureau_balance_features(
        application_train: pd.DataFrame,
        bureau: pd.DataFrame,
        bureau_balance: pd.DataFrame,
) -> pd.DataFrame:
    bureau_balance_features = build_bureau_balance_agg_features(
        bureau=bureau,
        bureau_balance=bureau_balance,
    )
    return application_train.merge(
        bureau_balance_features,
        on="SK_ID_CURR",
        how="left",
    )


def build_credit_card_features(
        credit_data: pd.DataFrame
) -> pd.DataFrame:
    credit_card_features = credit_data.copy()
    credit_card_features['balance_to_limit'] = (
            credit_card_features['AMT_BALANCE'] /
            credit_card_features['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
    )
    credit_card_features['payment_to_min'] = credit_card_features['AMT_PAYMENT_CURRENT'] / credit_card_features[
        'AMT_INST_MIN_REGULARITY'].replace(0, np.nan)
    credit_card_features['payment_to_balance'] = credit_card_features['AMT_PAYMENT_CURRENT'] / credit_card_features[
        'AMT_BALANCE'].replace(0, np.nan)
    credit_card_features['drawings_to_limit'] = credit_card_features['AMT_DRAWINGS_CURRENT'] / credit_card_features[
        'AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
    cc_agg = credit_card_features.groupby('SK_ID_CURR').agg(
        balance_to_limit_mean=('balance_to_limit', 'mean'),
        balance_to_limit_max=('balance_to_limit', 'max'),
        dpd_mean=('SK_DPD', 'mean'),
        dpd_max=('SK_DPD', 'max'),
        payment_to_min_mean=('payment_to_min', 'mean'),
        payment_to_min_max=('payment_to_min', 'max'),
        payment_to_balance_mean=('payment_to_balance', 'mean'),
        payment_to_balance_max=('payment_to_balance', 'max'),
        drawings_to_limit_mean=('drawings_to_limit', 'mean'),
        drawings_to_limit_max=('drawings_to_limit', 'max')
    ).reset_index()
    return cc_agg


def merge_credit_credit_features(application_train: pd.DataFrame, credit_card_balance: pd.DataFrame) -> pd.DataFrame:
    credit_card_features = build_credit_card_features(credit_card_balance)
    return application_train.merge(credit_card_features, on="SK_ID_CURR", how="left")


def build_installments_payments_features(
        credit_data: pd.DataFrame
) -> pd.DataFrame:
    ins_pay = credit_data
    ins_pay['days_late'] = ins_pay['DAYS_ENTRY_PAYMENT'] - ins_pay['DAYS_INSTALMENT']
    ins_pay['late_flag'] = (ins_pay['days_late'] > 0).astype(int)
    ins_pay['pay_diff'] = ins_pay['AMT_PAYMENT'] - ins_pay['AMT_INSTALMENT']
    ins_pay['pay_ratio'] = ins_pay['AMT_PAYMENT'] / ins_pay['AMT_INSTALMENT'].replace(0, np.nan)
    ins_pay['early_flag'] = (ins_pay['days_late'] < 0).astype(int)
    ins_pay['underpaid_flag'] = (ins_pay['pay_diff'] < 0).astype(int)
    ins_pay['overpaid_flag'] = (ins_pay['pay_diff'] > 0).astype(int)
    ins_pay['severe_late_30'] = (ins_pay['days_late'] > 30).astype(int)
    ins_agg = ins_pay.groupby('SK_ID_CURR').agg(
        INS_DAYS_LATE_MEAN=('days_late', 'mean'),
        INS_DAYS_LATE_MAX=('days_late', 'max'),
        INS_LATE_RATIO=('late_flag', 'mean'),
        INS_LATE_COUNT=('late_flag', 'sum'),
        INS_EARLY_RATIO=('early_flag', 'mean'),
        INS_PAY_DIFF_MEAN=('pay_diff', 'mean'),
        INS_PAY_DIFF_MIN=('pay_diff', 'min'),
        INS_PAY_RATIO_MEAN=('pay_ratio', 'mean'),
        INS_PAY_RATIO_MIN=('pay_ratio', 'min'),
        INS_UNDERPAID_RATIO=('underpaid_flag', 'mean'),
        INS_SEVERE_LATE_30_RATIO=('severe_late_30', 'mean'),
        INS_AMT_PAYMENT_SUM=('AMT_PAYMENT', 'sum'),
        INS_AMT_INSTALMENT_SUM=('AMT_INSTALMENT', 'sum'),
        INS_RECORD_COUNT=('SK_ID_PREV', 'count'),
        INS_PREV_LOAN_NUNIQUE=('SK_ID_PREV', 'nunique')
    ).reset_index()
    return ins_agg


def merge_installments_payments_features(application_train: pd.DataFrame,
                                         installments_payments: pd.DataFrame) -> pd.DataFrame:
    installments_payments_features = build_installments_payments_features(installments_payments)
    return application_train.merge(installments_payments_features, on="SK_ID_CURR", how="left")


def build_POS_CASH_features(POS_CASH_balance: pd.DataFrame) -> pd.DataFrame:
    pcb = POS_CASH_balance.copy()
    pcb['DPD_POS_FLAG'] = (pcb['SK_DPD'] > 0).astype(int)
    pcb['DPD_DEF_POS_FLAG'] = (pcb['SK_DPD_DEF'] > 0).astype(int)
    pcb['STATUS_COMPLETED_FLAG'] = (pcb['NAME_CONTRACT_STATUS'] == 'Completed').astype(int)
    pcb_agg_sk_id_prev = pcb.groupby('SK_ID_PREV').agg(
        SK_ID_CURR=('SK_ID_CURR', 'first'),
        PCD_PREV_MONTH_MAX=('MONTHS_BALANCE', 'max'),
        PCD_PREV_MONTH_MIN=('MONTHS_BALANCE', 'min'),
        PCD_PREV_MONTH_RECORDS=('SK_ID_PREV', 'count'),
        PCD_CNT_INSTALLMENTS_MAX=('CNT_INSTALMENT', 'max'),
        PCD_CNT_INSTALLMENTS_MIN=('CNT_INSTALMENT', 'min'),
        PCD_CNT_INSTALLMENTS_MEAN=('CNT_INSTALMENT', 'mean'),
        PCD_CNT_INSTALMENT_FUTURE_max=('CNT_INSTALMENT_FUTURE', 'max'),
        PCD_CNT_INSTALMENT_FUTURE_mean=('CNT_INSTALMENT_FUTURE', 'mean'),
        PCD_CNT_INSTALMENT_FUTURE_min=('CNT_INSTALMENT_FUTURE', 'min'),
        PREV_DPD_MAX=('SK_DPD', 'max'),
        PREV_DPD_MEAN=('SK_DPD', 'mean'),
        PREV_DPD_SUM=('SK_DPD', 'sum'),
        PREV_DPD_DEF_MAX=('SK_DPD_DEF', 'max'),
        PREV_DPD_DEF_MEAN=('SK_DPD_DEF', 'mean'),
        PREV_DPD_DEF_SUM=('SK_DPD_DEF', 'sum'),
        PREV_LATE_MONTHS_COUNT=('DPD_POS_FLAG', 'sum'),
        PREV_LATE_DEF_MONTHS_COUNT=('DPD_DEF_POS_FLAG', 'sum'),
        PREV_STATUS_NUNIQUE=('NAME_CONTRACT_STATUS', 'nunique'),
        PREV_WAS_COMPLETED=('STATUS_COMPLETED_FLAG', 'max')
    ).reset_index()
    pcb_agg_sk_id_curr = pcb_agg_sk_id_prev.groupby('SK_ID_CURR').agg(
        POS_PREV_CREDIT_COUNT=('SK_ID_PREV', 'nunique'),

        POS_PREV_MONTH_RECORDS_SUM=('PCD_PREV_MONTH_RECORDS', 'sum'),
        POS_PREV_MONTH_RECORDS_MEAN=('PCD_PREV_MONTH_RECORDS', 'mean'),

        POS_CNT_INSTALLMENTS_MAX=('PCD_CNT_INSTALLMENTS_MAX', 'max'),
        POS_CNT_INSTALLMENTS_MEAN=('PCD_CNT_INSTALLMENTS_MEAN', 'mean'),

        POS_CNT_INSTALMENT_FUTURE_MAX=('PCD_CNT_INSTALMENT_FUTURE_max', 'max'),
        POS_CNT_INSTALMENT_FUTURE_MEAN=('PCD_CNT_INSTALMENT_FUTURE_mean', 'mean'),
        POS_CNT_INSTALMENT_FUTURE_MIN=('PCD_CNT_INSTALMENT_FUTURE_min', 'min'),

        POS_DPD_MAX=('PREV_DPD_MAX', 'max'),
        POS_DPD_MEAN=('PREV_DPD_MEAN', 'mean'),
        POS_DPD_SUM=('PREV_DPD_SUM', 'sum'),

        POS_DPD_DEF_MAX=('PREV_DPD_DEF_MAX', 'max'),
        POS_DPD_DEF_MEAN=('PREV_DPD_DEF_MEAN', 'mean'),
        POS_DPD_DEF_SUM=('PREV_DPD_DEF_SUM', 'sum'),

        POS_LATE_MONTHS_COUNT_SUM=('PREV_LATE_MONTHS_COUNT', 'sum'),
        POS_LATE_DEF_MONTHS_COUNT_SUM=('PREV_LATE_DEF_MONTHS_COUNT', 'sum'),

        POS_STATUS_NUNIQUE_MAX=('PREV_STATUS_NUNIQUE', 'max'),
        POS_COMPLETED_CREDIT_COUNT=('PREV_WAS_COMPLETED', 'sum'),
        POS_COMPLETED_CREDIT_RATIO=('PREV_WAS_COMPLETED', 'mean')
    ).reset_index()
    return pcb_agg_sk_id_curr


def merge_POS_CASH_BALANCE_features(application_train: pd.DataFrame,
                                    POS_CASH_BALANCE: pd.DataFrame) -> pd.DataFrame:
    POS_CASH_BALANCE_features = build_POS_CASH_features(POS_CASH_BALANCE)
    return application_train.merge(POS_CASH_BALANCE_features, on="SK_ID_CURR", how="left")

def build_previous_application_features(previous_application: pd.DataFrame) -> pd.DataFrame:
    prev = previous_application.copy()

    prev['PREV_APPROVED_FLAG'] = (prev['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    prev['PREV_REFUSED_FLAG'] = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    prev['PREV_CANCELED_FLAG'] = (prev['NAME_CONTRACT_STATUS'] == 'Canceled').astype(int)
    prev['PREV_UNUSED_FLAG'] = (prev['NAME_CONTRACT_STATUS'] == 'Unused offer').astype(int)

    prev['PREV_CREDIT_APP_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['PREV_CREDIT_APP_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_APPLICATION']

    prev_features = prev.groupby('SK_ID_CURR').agg(
        PREV_APPLICATION_COUNT=('SK_ID_PREV', 'nunique'),
        PREV_AMT_APPLICATION_MEAN=('AMT_APPLICATION', 'mean'),
        PREV_AMT_APPLICATION_MAX=('AMT_APPLICATION', 'max'),
        PREV_AMT_CREDIT_MEAN=('AMT_CREDIT', 'mean'),
        PREV_AMT_CREDIT_MAX=('AMT_CREDIT', 'max'),
        PREV_AMT_ANNUITY_MEAN=('AMT_ANNUITY', 'mean'),
        PREV_AMT_ANNUITY_MAX=('AMT_ANNUITY', 'max'),
        PREV_CNT_PAYMENT_MEAN=('CNT_PAYMENT', 'mean'),
        PREV_CNT_PAYMENT_MAX=('CNT_PAYMENT', 'max'),
        PREV_DAYS_DECISION_MAX=('DAYS_DECISION', 'max'),
        PREV_DAYS_DECISION_MIN=('DAYS_DECISION', 'min'),
        PREV_CREDIT_APP_DIFF_MEAN=('PREV_CREDIT_APP_DIFF', 'mean'),
        PREV_CREDIT_APP_RATIO_MEAN=('PREV_CREDIT_APP_RATIO', 'mean'),
        PREV_APPROVED_COUNT=('PREV_APPROVED_FLAG', 'sum'),
        PREV_REFUSED_COUNT=('PREV_REFUSED_FLAG', 'sum'),
        PREV_CANCELED_COUNT=('PREV_CANCELED_FLAG', 'sum'),
        PREV_UNUSED_COUNT=('PREV_UNUSED_FLAG', 'sum'),
        PREV_APPROVED_RATIO=('PREV_APPROVED_FLAG', 'mean'),
        PREV_REFUSED_RATIO=('PREV_REFUSED_FLAG', 'mean'),
    ).reset_index()

    approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved'].groupby('SK_ID_CURR').agg(
        APPROVED_AMT_CREDIT_MEAN=('AMT_CREDIT', 'mean'),
        APPROVED_AMT_CREDIT_MAX=('AMT_CREDIT', 'max'),
        APPROVED_AMT_ANNUITY_MEAN=('AMT_ANNUITY', 'mean'),
        APPROVED_CNT_PAYMENT_MEAN=('CNT_PAYMENT', 'mean'),
        APPROVED_DAYS_DECISION_MAX=('DAYS_DECISION', 'max'),
    ).reset_index()

    prev_features = prev_features.merge(approved, on='SK_ID_CURR', how='left')

    return prev_features

def merge_previous_application_features(application_train: pd.DataFrame,
                                    previous_application: pd.DataFrame) -> pd.DataFrame:
    previous_applicationE_features = build_previous_application_features(previous_application)
    return application_train.merge(previous_applicationE_features, on="SK_ID_CURR", how="left")