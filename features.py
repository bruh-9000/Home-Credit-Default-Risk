import numpy as np
import pandas as pd

def add_custom_features(main_df, bureau_df, bureau_balance_df, previous_application_df):
    # Bureau Balance Features
    bur_bal_grouped = bureau_balance_df.groupby('SK_ID_BUREAU')
    bureau_balance_features = pd.concat([
        bur_bal_grouped['MONTHS_BALANCE'].min().rename('OLDEST_BALANCE_MONTH'),
        bur_bal_grouped['STATUS'].agg(lambda x: 'C' in set(x)).astype(int).rename('EVER_C')
    ], axis=1).reset_index()
    bureau_df = bureau_df.merge(bureau_balance_features, on='SK_ID_BUREAU', how='left')


    # Previous App Features
    yield_score_map = {
        'low_action': 1, 'low_normal': 2, 'middle': 3, 'high': 4, 'XNA': 0
    }
    upper = previous_application_df['AMT_DOWN_PAYMENT'].quantile(0.90)
    previous_application_df['AMT_DOWN_PAYMENT'] = previous_application_df['AMT_DOWN_PAYMENT'].clip(upper=upper)
    prev_grouped = previous_application_df.groupby('SK_ID_CURR')
    previous_application_df['CNT_CONTSTATUS_APPROVED'] = (previous_application_df['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    previous_application_df['CNT_CONTSTATUS_REFUSED'] = (previous_application_df['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    previous_application_df['CNT_CONTSTATUS_CANCELED'] = (previous_application_df['NAME_CONTRACT_STATUS'] == 'Canceled').astype(int)
    previous_application_df['CNT_REPEAT_CLIENT'] = (previous_application_df['NAME_CLIENT_TYPE'] == 'Repeater').astype(int)
    previous_application_df['YIELD_SCORE_MEAN'] = previous_application_df['NAME_YIELD_GROUP'].map(yield_score_map)
    prev_features = pd.concat([
        prev_grouped['AMT_DOWN_PAYMENT'].mean().rename('AMT_DOWN_PAYMENT_MEAN'),
        prev_grouped['CNT_CONTSTATUS_APPROVED'].sum(),
        prev_grouped['CNT_CONTSTATUS_REFUSED'].sum(),
        prev_grouped['CNT_CONTSTATUS_CANCELED'].sum(),
        prev_grouped['CNT_REPEAT_CLIENT'].sum(),
        prev_grouped['YIELD_SCORE_MEAN'].mean(),
        prev_grouped['NFLAG_INSURED_ON_APPROVAL'].mean().rename('INSURED_MEAN')
    ], axis=1).reset_index()


    # Bureau Features
    bur_grouped = bureau_df.groupby('SK_ID_CURR')
    bureau_df['CREDIT_CLOSED_CNT'] = (bureau_df['CREDIT_ACTIVE'] == 'Closed').astype(int)
    bureau_df['CREDIT_ACTIVE_CNT'] = (bureau_df['CREDIT_ACTIVE'] == 'Active').astype(int)
    bureau_features = pd.concat([
        bur_grouped['CREDIT_CLOSED_CNT'].sum(),
        bur_grouped['CREDIT_ACTIVE_CNT'].sum(),
        bur_grouped['DAYS_CREDIT'].mean().rename('DAYS_CREDIT_MEAN'),
        bur_grouped['DAYS_CREDIT_ENDDATE'].mean().rename('DAYS_CREDIT_ENDDATE_MEAN'),
        bur_grouped['AMT_CREDIT_SUM'].mean().rename('AMT_CREDIT_SUM_MEAN'),
        bur_grouped['AMT_CREDIT_SUM_DEBT'].mean().rename('AMT_CREDIT_SUM_DEBT_MEAN'),
        bur_grouped['AMT_CREDIT_SUM_LIMIT'].mean().rename('AMT_CREDIT_SUM_LIMIT_MEAN'),
        bur_grouped['DAYS_CREDIT_UPDATE'].mean().rename('DAYS_CREDIT_UPDATE_MEAN'),
        bur_grouped['OLDEST_BALANCE_MONTH'].mean().rename('OLDEST_BALANCE_MONTH_MEAN'),
        bur_grouped['EVER_C'].mean().rename('EVER_C_PERCENT')
    ], axis=1).reset_index()


    # Merge
    main_df = main_df.merge(bureau_features, on='SK_ID_CURR', how='left')
    main_df = main_df.merge(prev_features, on='SK_ID_CURR', how='left')


    # Synth Features
    main_df['NAME_FAMILY_STATUS_Married'] = main_df['NAME_FAMILY_STATUS'] == 'Married'

    doc_flags = [f'FLAG_DOCUMENT_{i}' for i in [2, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
    main_df['ANY_DOC_FLAGGED'] = main_df[doc_flags].any(axis=1)

    main_df['HEAVILY_OVERFINANCED'] = main_df['AMT_CREDIT'] > main_df['AMT_GOODS_PRICE'] * 1.13
    main_df['EXPECTED_TERM'] = main_df['AMT_CREDIT'] / main_df['AMT_ANNUITY']


    return main_df