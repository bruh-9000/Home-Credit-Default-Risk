This experiment tracker was started in the middle of the project. Prior changes are not reflected in these notes

# 1

Roc_auc: 77.28% ± 0.18%

# 2

Testing adding ANY_DOC_FLAGGED. Comparing against #1's Roc_auc: 77.28% ± 0.18%
Roc_auc: 77.26% ± 0.20%
Adds more noise and lower mean overall, removing the feature

# 3

Submitted lightgbm pipeline, got .77450 public and .76937 private. Best score so far

# 4

Testing removal of CNT_CONTSTATUS_CANCELLED. Comparing against #1's Roc_auc: 77.28% ± 0.18%
Roc_auc: 77.24% ± 0.19%
Reverting changes

# 5

Testing encoding NAME_HOUSING_TYPE as freq not target. Comparing against #1's Roc_auc: 77.28% ± 0.18%
Roc_auc: 77.26% ± 0.21%
Reverting changes

# 6

Testing OWN_CAR_AGE imputation as -1, was 0, and AMT_DOWN_PAYMENT_MEAN imputation as mean, was 1. Comparing against #1's Roc_auc: 77.28% ± 0.18%
Roc_auc: 77.27% ± 0.19%
While it technically lowered scores very minimally, it likely helped generalization. Keeping changes

# 7

Testing removing NAME_HOUSING_TYPE and replacing with IS_HOUSE_APT as bool. NAME_HOUSING_TYPE is 88.8% one value. Previous SHAP importance was 0.022. Comparing against #6's Roc_auc: 77.27% ± 0.19%
Roc_auc: 77.25% ± 0.20%, SHAP importance dropped to 0.014
Reverting changes

# 8

Testing DAYS_EMPLOYED's 365243 values (missing) imputed as mean. Comparing against #6's Roc_auc: 77.27% ± 0.19%
Roc_auc: 77.28% ± 0.18%
Improved mean AUC by 0.01% and lowered std by 0.01%. Likely noise, and model probably generalizes better with 365243 as a missing flag. Reverting changes

# 9

Submitted lightgbm pipeline, got 0.77692 public and 0.76922 private. New best on public, second best on private

# 10

Testing clipping DAYS_CREDIT_ENDDATE_MEAN to 99th percentile, max was 31198 which is abnormally high. Comparing against #6's Roc_auc: 77.27% ± 0.19%
Roc_auc: 77.24% ± 0.19%
Reverting changes

# 11

Testing imputating DAYS_FIRST_DUE_MEAN's 365243 values (missing) as mean. Comparing against #6's Roc_auc: 77.27% ± 0.19%
Roc_auc: 77.26% ± 0.18%
Reverting changes

# 12

Testing adding new features from POS_CASH_balance.csv and installments_payments.csv. Comparing against #6's Roc_auc: 77.27% ± 0.19%
Roc_auc: 77.64% ± 0.18%
Significant gain, keeping changes

# 13

Testing changing SK_DPD_MEAN to MAX. Comparing against #12's Roc_auc: 77.64% ± 0.18%
Roc_auc: 77.66% ± 0.19%
Keeping changes

# 14

Submitted lightgbm pipeline, got 0.77807 public and 0.77362 private. Best score so far

# 15

Testing changing resampling strategy to .45 from .5. Comparing against #13's Roc_auc: 77.66% ± 0.19%
Roc_auc: 77.65% ± 0.12%
While the mean AUC dropped by 0.01%, that's likely noise. The std drop by 0.07% is significant, though. Keeping changes

# 16

Testing changing resampling strategy to .4 from .45
Roc_auc: 77.71% ± 0.12%
AUC jumped again. Keeping changes

# 17

Since the last two lowerings worked well, changing resampling strategy to .2 from .4. Comparing against #15's Roc_auc: 77.65% ± 0.12%
Roc_auc: 77.79% ± 0.20%
While the mean is higher, the std is also higher. Reverting changes

# 18

Testing resampling strategy of .3. Comparing against #15's Roc_auc: 77.65% ± 0.12%
Roc_auc: 77.75% ± 0.14%
This is more stable. Slightly lower mean, but significantly lower std. Keeping changes

# 19

Submitted lightgbm pipeline, got 0.77721 public and 0.77580 private. New best on private, second best on public

# 20

Testing removal of PAYMENT_DIFF_STD, lowest SHAP importance at 0.011. Comparing against #18's Roc_auc: 77.75% ± 0.14%
Roc_auc: 77.75% ± 0.17%
Increased std but not mean. Reverting changes

# 21

Testing adding CREDIT_TYPE_CONSUMER_SUM, CREDIT_TYPE_NOT_CONSUMER_SUM from bureau.csv. Comparing against #18's Roc_auc: 77.75% ± 0.14%
Roc_auc: 77.81% ± 0.14%
Mean increased by 0.06%, std stayed the same. Keeping changes

# 22

Testing changing DAYS_CREDIT_UPDATE from mean to min, clipping at 1 and 99 due to outliers. Comparing against #21's Roc_auc: 77.81% ± 0.14%
Roc_auc: 77.80% ± 0.15%
While it dropped slightly in performance, this likely will help generalization. Keeping changes

# 23

Testing adding TOP_GOODS_CATEGORY as target, with impution method of prior. Feature std is 0.0123. Comparing against #22's Roc_auc: 77.80% ± 0.15%
Roc_auc: 77.78% ± 0.18%
Mean AUC dropped. Reverting changes

# 24

Testing impution for TOP_GOODS_CATEGORY as 'Unknown'. Feature std is 0.0135. Comparing against #22's Roc_auc: 77.80% ± 0.15%
Roc_auc: 77.84% ± 0.16%
Mean increased. Keeping changes

# 25

Testing TOP_GOODS_CATEGORY by combining low freq categories at threshold 5% into 'Other'. Feature std is 0.0123. Comparing against #24's Roc_auc: 77.84% ± 0.16%
Roc_auc: 77.79% ± 0.15%
Mean dropped by 0.05%. Reverting changes

# 26

Testing TOP_GOODS_CATEGORY merge threshold at 1%. Feature std is 0.0130. Comparing against #24's Roc_auc: 77.84% ± 0.16%
Roc_auc: 77.84% ± 0.18%
Std increased. Reverting changes

# 27

Testing impution for TOP_GOODS_CATEGORY as 'XNA' (existing unknown category). Feature std is 0.0119. Comparing against #24's Roc_auc: 77.84% ± 0.16%
Roc_auc: 77.82% ± 0.19%
Reverting changes

# 28

Testing imputating four features as mean instead of -1. Comparing against #24's Roc_auc: 77.84% ± 0.16%
Roc_auc: 77.81% ± 0.13%
Reverting changes

# 29

Testing OCCUPATION_TYPE merge threshold at .75%. Prev std was 0.0219, now is 0.0208. Comparing against #24's Roc_auc: 77.84% ± 0.16%
Roc_auc: 77.81% ± 0.19%
Reverting changes

# 30

Testing imputation for NAME_HOUSING_TYPE as mode instead of prior. Comparing against #24's Roc_auc: 77.84% ± 0.16%
Roc_auc: 77.84% ± 0.16%
No change in stats, but likely helps the model generalize better. Keeping changes

# 31

Submitted lightgbm pipeline, got 0.77779 public and 0.77728 private. New best on private, second best on public

# 32

Testing replacing REGION_RATING_CLIENT with REGION_RATING_CLIENT_W_CITY. Comparing against #30's Roc_auc: 77.84% ± 0.16%
Roc_auc: 77.80% ± 0.15%
Reverting changes

# 33

Testing removing FLAG_OWN_CAR, OWN_CAR_AGE already imputates with -1. Comparing against #30's Roc_auc: 77.84% ± 0.16%
Roc_auc: 77.83% ± 0.16%
Reverting changes

# 34

Testing adding TERM_COST, EXT_SOURCE_MEAN, EXT_SOURCE_STD. Comparing against #30's Roc_auc: 77.84% ± 0.16%
Roc_auc: 77.84% ± 0.16%
Reverting changes

# 35

Testing adding TERM_COST, EXT_SOURCE_STD. Comparing against #30's Roc_auc: 77.84% ± 0.16%
Roc_auc: 77.76% ± 0.19%
Reverting changes

# 36

Testing adding TERM_COST. Comparing against #30's Roc_auc: 77.84% ± 0.16%
Roc_auc: 77.79% ± 0.17%
Looks like none of the three synth features gave any new insight for the model. Reverting changes

# 37

Note: At Roc_auc: 77.81% ± 0.16%, unknown which change dropped it. With reverting config, I am unable to return it to 77.84%

# 38

Testing changing DAYS_LAST_PHONE_CHANGE imputation to mode, CNT_FAM_MEMBERS imputation to median, cast to int. Comparing to #37's Roc_auc: 77.81% ± 0.16%
Roc_auc: 77.82% ± 0.13%
Mean and std improved. Keeping changes

# 39

Testing adding AGG_MISSING_COUNT as many of the agg features from the non-main files are missing for users. Comparing to #38's Roc_auc: 77.82% ± 0.13%
Roc_auc: 77.80% ± 0.18%
Reverting changes

# 40

Testing AGG_MISSING_COUNT without using OLDEST_BALANCE_MONTH_MIN because it had a high corr with AGG_MISSING_COUNT. Comparing to #38's Roc_auc: 77.82% ± 0.13%
Roc_auc: 77.80% ± 0.19%
Reverting changes

# 41

Testing changing CNT_CONTSTATUS_CANCELED imputation to 0. Comparing to #38's Roc_auc: 77.82% ± 0.13%
Roc_auc: 77.82% ± 0.13%
No change in performance, but likely is a good change. Keeping changes

# 42

Testing adding CNT_CHILDREN, INCOME_PER_KID as features. Comparing to #41's Roc_auc: 77.82% ± 0.13%
Roc_auc: 77.81% ± 0.18%
INCOME_PER_KID performed better than CNT_CHILDREN. Reverting changes

# 43

Testing adding INCOME_PER_KID. Comparing to #41's Roc_auc: 77.82% ± 0.13%
Roc_auc: 77.80% ± 0.22%
Reverting changes

# 44

Testing imputating EXT_SOURCE_1 as -1. It has 45% missing values, so mean might be removing some signal that would otherwise be shown. Comparing to #41's Roc_auc: 77.82% ± 0.13%
Roc_auc: 77.85% ± 0.13%
Improved AUC, and moved EXT_SOURCE_1 to 3rd place in SHAP importance. Keeping changes

# 45

Testing imputating EXT_SOURCE_2 and EXT_SOURCE_3 as -1. Comparing to #44's Roc_auc: 77.85% ± 0.13%
Roc_auc: 77.86% ± 0.14%
EXT_SOURCE_2 swapped SHAP importance with EXT_SOURCE_3. Keeping changes

# 46

Testing imputating EXPECTED_TERM and OVERFINANCE_AMOUNT as -1. Comparing to #45's Roc_auc: 77.86% ± 0.14%
Roc_auc: 77.89% ± 0.21%
While the mean went up by 0.03%, std jumped by 0.07%. Reverting changes

# 47

Testing removing PAYMENT_DIFF_STD due to high corr with PAYMENT_DIFF_MEAN, and it having significantly lower SHAP importance of the two. Comparing to #45's Roc_auc: 77.86% ± 0.14%
Roc_auc: 77.87% ± 0.17%
Reverting changes

# 48

Testing imputating LATE_DAYS_MEAN as 0. LATE_DAYS_MAX is already imputated as 0, and likely both will be missing together. Comparing to #45's Roc_auc: 77.86% ± 0.14%
Roc_auc: 77.85% ± 0.16%
Reverting changes

# 49

Testing imputating PROFIT_AMOUNT as 1 instead of mean. Comparing to #45's Roc_auc: 77.86% ± 0.14%
Roc_auc: 77.87% ± 0.18%
Reverting changes

# 50

Submitted lightgbm pipeline, got 0.77880 public and 0.77504 private. New best on public, third best on private. Possible overfitting

# 51

Testing adding CREDIT_CLOSED_CNT_SUM, CREDIT_ACTIVE_CNT_SUM, CNT_CONTSTATUS_APPROVED_SUM, CNT_CONTSTATUS_REFUSED_SUM, CNT_CONTSTATUS_CANCELED_SUM. Comparing to #45's Roc_auc: 77.86% ± 0.14%
Roc_auc: 77.97% ± 0.18%
Keeping changes

# 52

Testing adding CREDIT_DAY_OVERDUE_MAX. Comparing to #51's Roc_auc: 77.97% ± 0.18%
Roc_auc: 77.93% ± 0.25%
Reverting changes

# 53

Testing adding CNT_PAYMENT_STD, AMT_CREDIT_SUM_DEBT_STD. Comparing to #51's Roc_auc: 77.97% ± 0.18%
Roc_auc: 78.03% ± 0.17%
Keeping changes

# 54

Testing removing CNT_CONTSTATUS_CANCELED_SUM due to its SHAP importance of 0.004. Comparing to #53's Roc_auc: 78.03% ± 0.17%
Roc_auc: 78.01% ± 0.19%
Reverting changes

# 55

Submitted lightgbm pipeline (on #54), got 0.78124 public and 0.77777 private. Best score so far

# 56

Testing performance increasing n_iter from 35 to 100, adding 450, 500, 550 to n_estimators. Comparing to #53's Roc_auc: 78.03% ± 0.17%
Roc_auc: 78.21% ± 0.14%

# 57

Submitted lightgbm pipeline, got 0.78272 public and 0.77945 private. Best score so far


# 58

Testing lightgbm performance, got Roc_auc: 78.16% ± 0.21%
