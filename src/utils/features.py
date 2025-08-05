import numpy as np


def get_bureau_features(bureau_df, bureau_balance_df):
    bur_bal = bureau_balance_df.groupby('SK_ID_BUREAU').agg({
        'MONTHS_BALANCE': 'min',
        'STATUS': lambda x: 'C' in set(x)
    }).rename(columns={
        'MONTHS_BALANCE': 'OLDEST_BALANCE_MONTH',
        'STATUS': 'EVER_C'
    }).reset_index()
    bur_bal['EVER_C'] = bur_bal['EVER_C'].astype(int)

    bureau_df = bureau_df.merge(bur_bal, on='SK_ID_BUREAU', how='left')
    bureau_df['CREDIT_CLOSED_CNT'] = (
        bureau_df['CREDIT_ACTIVE'] == 'Closed').astype(int)
    bureau_df['CREDIT_ACTIVE_CNT'] = (
        bureau_df['CREDIT_ACTIVE'] == 'Active').astype(int)
    bureau_df['CREDIT_TYPE_CONSUMER'] = (
        bureau_df['CREDIT_TYPE'] == 'Consumer credit').astype(int)
    bureau_df['CREDIT_TYPE_NOT_CONSUMER'] = (
        bureau_df['CREDIT_TYPE'] != 'Consumer credit').astype(int)

    agg = bureau_df.groupby('SK_ID_CURR').agg({
        'CREDIT_CLOSED_CNT': ['sum'],
        'CREDIT_ACTIVE_CNT': ['sum'],
        'DAYS_CREDIT': ['mean'],
        'DAYS_CREDIT_ENDDATE': ['mean'],
        'DAYS_ENDDATE_FACT': ['mean'],
        'AMT_CREDIT_SUM': ['mean'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'std'],
        'AMT_CREDIT_SUM_LIMIT': ['mean'],
        'CREDIT_TYPE_CONSUMER': ['sum'],
        'CREDIT_TYPE_NOT_CONSUMER': ['sum'],
        'DAYS_CREDIT_UPDATE': ['min'],
        'CREDIT_DAY_OVERDUE': ['max'],
        'OLDEST_BALANCE_MONTH': ['min'],
        'EVER_C': ['mean']
    })
    agg.columns = [f"{col[0]}_{col[1].upper()}" for col in agg.columns]
    agg = agg.reset_index()

    return agg


def get_previous_app_features(prev_df):
    yield_map = {'low_action': 1, 'low_normal': 2,
                 'middle': 3, 'high': 4, 'XNA': 0}
    prev_df['YIELD_SCORE'] = prev_df['NAME_YIELD_GROUP'].map(yield_map)
    prev_df['APPLICATION_RATIO'] = prev_df['AMT_APPLICATION'] / \
        prev_df['AMT_CREDIT']
    prev_df['APPLICATION_RATIO'] = prev_df['APPLICATION_RATIO'].replace(
        [np.inf, -np.inf], np.nan).fillna(1)

    upper = prev_df['AMT_DOWN_PAYMENT'].quantile(0.90)
    prev_df['AMT_DOWN_PAYMENT'] = prev_df['AMT_DOWN_PAYMENT'].clip(upper=upper)

    prev_df['CNT_CONTSTATUS_APPROVED'] = (
        prev_df['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    prev_df['CNT_CONTSTATUS_REFUSED'] = (
        prev_df['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    prev_df['CNT_CONTSTATUS_CANCELED'] = (
        prev_df['NAME_CONTRACT_STATUS'] == 'Canceled').astype(int)

    top_category = (
        prev_df.groupby(['SK_ID_CURR', 'NAME_GOODS_CATEGORY'])
        .size()
        .reset_index(name='count')
        .sort_values(['SK_ID_CURR', 'count'], ascending=[True, False])
        .drop_duplicates('SK_ID_CURR')
        .rename(columns={'NAME_GOODS_CATEGORY': 'TOP_GOODS_CATEGORY'})
        [['SK_ID_CURR', 'TOP_GOODS_CATEGORY']]
    )

    agg = prev_df.groupby('SK_ID_CURR').agg({
        'AMT_DOWN_PAYMENT': ['mean'],
        'CNT_CONTSTATUS_APPROVED': ['sum'],
        'CNT_CONTSTATUS_REFUSED': ['sum'],
        'CNT_CONTSTATUS_CANCELED': ['sum'],
        'YIELD_SCORE': ['mean'],
        'NFLAG_INSURED_ON_APPROVAL': ['mean'],
        'AMT_APPLICATION': ['mean'],
        'APPLICATION_RATIO': ['mean'],
        'RATE_DOWN_PAYMENT': ['mean'],
        'CNT_PAYMENT': ['mean', 'std'],
        'DAYS_FIRST_DUE': ['mean'],
    })
    agg.columns = [f"{col[0]}_{col[1].upper()}" for col in agg.columns]
    agg = agg.reset_index()
    agg = agg.merge(top_category, on='SK_ID_CURR', how='left')

    return agg


def get_credit_card_features(cc_df):
    cc_df['BALANCE_LIMIT_RATIO'] = cc_df['AMT_BALANCE'] / \
        cc_df['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
    cc_df['PAYMENT_OVER_MIN'] = cc_df['AMT_PAYMENT_CURRENT'] - \
        cc_df['AMT_INST_MIN_REGULARITY']

    agg = cc_df.groupby('SK_ID_CURR').agg({
        'AMT_CREDIT_LIMIT_ACTUAL': ['mean'],
        'AMT_BALANCE': ['mean'],
        'CNT_INSTALMENT_MATURE_CUM': ['max'],
        'BALANCE_LIMIT_RATIO': ['mean']
    })
    agg.columns = [f"{col[0]}_{col[1].upper()}" for col in agg.columns]
    agg = agg.reset_index()

    return agg


def get_install_features(install_df):
    install_df['PAYMENT_DIFF'] = install_df['AMT_PAYMENT'] - \
        install_df['AMT_INSTALMENT']
    install_df['LATE_DAYS'] = install_df['DAYS_ENTRY_PAYMENT'] - \
        install_df['DAYS_INSTALMENT']

    agg = install_df.groupby('SK_ID_CURR').agg({
        'PAYMENT_DIFF': ['mean', 'std'],
        'LATE_DAYS': ['mean', 'max'],
        'AMT_PAYMENT': ['sum'],
    })
    agg.columns = [f"{col[0]}_{col[1].upper()}" for col in agg.columns]
    agg = agg.reset_index()

    return agg


def get_pos_cash_features(pos_cash_df):
    agg = pos_cash_df.groupby('SK_ID_CURR').agg({
        'SK_DPD': ['mean'],
        'SK_DPD_DEF': ['max'],
        'MONTHS_BALANCE': ['min', 'max', 'count'],
    })
    agg.columns = [f"{col[0]}_{col[1].upper()}" for col in agg.columns]
    agg = agg.reset_index()

    return agg


def add_features(main_df, bureau_df, bureau_balance_df, prev_df, cc_df, install_df, pos_cash_df):
    bureau_features = get_bureau_features(bureau_df, bureau_balance_df)
    prev_features = get_previous_app_features(prev_df)
    cc_features = get_credit_card_features(cc_df)
    install_features = get_install_features(install_df)
    pos_cash_features = get_pos_cash_features(pos_cash_df)

    main_df = main_df.merge(bureau_features, on='SK_ID_CURR', how='left')
    main_df = main_df.merge(prev_features, on='SK_ID_CURR', how='left')
    main_df = main_df.merge(cc_features, on='SK_ID_CURR', how='left')
    main_df = main_df.merge(install_features, on='SK_ID_CURR', how='left')
    main_df = main_df.merge(pos_cash_features, on='SK_ID_CURR', how='left')

    # Synthetic features

    # Whether the applicant is married (binary flag)
    main_df['NAME_FAMILY_STATUS_Married'] = (main_df['NAME_FAMILY_STATUS'] == 'Married')

    # Estimated overfinance amount, for if the loan is larger than the cost of the goods
    main_df['OVERFINANCE_AMOUNT'] = main_df['AMT_CREDIT'] / main_df['AMT_GOODS_PRICE']

    # Estimated term length of the loan
    main_df['EXPECTED_TERM'] = main_df['AMT_CREDIT'] / main_df['AMT_ANNUITY']

    # The estimated amount of money that will be needed to repay the loan, including interest
    main_df['EST_TOTAL_REPAID'] = main_df['AMT_ANNUITY'] * main_df['CNT_PAYMENT_MEAN']

    # The amount of profit that the loan provider is estimated to make
    main_df['PROFIT_AMOUNT'] = main_df['YIELD_SCORE_MEAN'] * main_df['AMT_CREDIT']

    # Income amount per year employed. Proxy for if raises and salary represents time spent working
    main_df['INCOME_PER_YEAR_EMPLOYED'] = main_df['AMT_INCOME_TOTAL'] / \
        (main_df['DAYS_EMPLOYED'].replace(365243, np.nan).abs().median() /
         365)  # 365243 is used for missing values, must be imputated

    return main_df
