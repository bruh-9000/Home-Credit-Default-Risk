# Core libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import joblib
import yaml

import pandas as pd
from IPython.display import display, Markdown
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    KFold,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)

from pathlib import Path
SAVE_DIR = Path(__file__).resolve().parent.parent / "saved"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Load config.yaml
ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Access values from the config
label = config["general"]["label"]
primary_metric = config["general"]["primary_metric"]
drop_features = config["preprocessing"]["drop_features"]
value_mappings = config["preprocessing"]["value_mappings"]
type_coercion = config["preprocessing"]["type_coercion"]
missing_handling = config["preprocessing"]["missing_handling"]

numerical_scale_cols = config["encoding"]["numerical_scale_cols"]
onehot_cols = config["encoding"]["onehot_cols"]
ordinal_cols = config["encoding"]["ordinal_cols"]

model_configs = config["model_configs"]

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error
)
import shap
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno



def load_data(file_one, file_two):
    return pd.read_csv(file_one), pd.read_csv(file_two) if file_two else None



def dedup(df):
    num_dupes = df.duplicated().sum()
    if num_dupes > 0:
        print(f"Successfully deleted {num_dupes} duplicated examples.")
        df = df.drop_duplicates()
    return df



def prepare_train_test_split(train_df, test_df, label):
    X = train_df
    y = train_df[label]

    # Case 1: Only one file (split into train/test)
    if test_df is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_val = y_val = None

    else:
        X_test = test_df
        X_train = train_df
        y_train = y

        # Case 2: If test has labels, use as test set
        if label in test_df.columns:
            y_test = test_df[label]
            X_val = y_val = None

        # Case 3: If test has NO labels, make val set from train
        else:
            y_test = None

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
            )

    return X, y, X_train, X_test, y_train, y_test, X_val, y_val



# Data preparation utilities



def show_categorical_uniques(df, limit=10):
    for col in df.select_dtypes(include=["object", "category"]).columns:
        uniques = df[col].unique()
        print(f"{col} ({len(uniques)} unique): {uniques[:limit]}")



def show_missing_data(df):
    # Check missing data percentages per column
    if df.isna().sum().sum() > 0:
        missing_counts = df.isna().sum()

        # Low missing amounts (per column)
        print('Missing values detected in:')
        print(missing_counts[missing_counts > 0])

        # Plot
        msno.matrix(df, figsize=(8.5, 4), fontsize=10)
        plt.title('Missing Data by Column')
        plt.show()
    else:
        print("No missing data found!")



# Feature engineering pipeline



# Drop columns based on config.py list
def drop_columns(X):
    existing = [col for col in drop_features if col in X.columns]
    return X.drop(columns=existing)
dropper = FunctionTransformer(drop_columns, feature_names_out='one-to-one')



# Map values for features (e.g., 'yes' -> 1, 'no' -> 0)
def apply_mappings(X):
    for col, mapping in value_mappings.items():
        if col != label and col in X.columns:
            X[col] = X[col].map(mapping)
    return X
mapper = FunctionTransformer(apply_mappings, feature_names_out='one-to-one')



# Make sure each feature is the correct type
def coerce_types(X):
    return X.astype(type_coercion)
coercer = FunctionTransformer(coerce_types, feature_names_out='one-to-one')



# Perform actions based on config.py setting for each feature
def handle_missing_values(X):
    if missing_handling is None:
        return X
    
    X = X.copy()
    for col, strategy in missing_handling.items():
        if col not in X.columns:
            continue
        if strategy == 'drop':
            X = X.dropna(subset=[col])
        elif strategy == 'mean':
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].mean())
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
        elif strategy == 'mode':
            mode_val = X[col].mode()
            if not mode_val.empty:
                X[col] = X[col].fillna(mode_val[0])
        elif strategy == 'prior':
            non_na = X[col].dropna()
            if non_na.nunique() == 2:
                probs = non_na.value_counts(normalize=True)
                X[col] = X[col].apply(
                    lambda x: np.random.choice(probs.index, p=probs.values) if pd.isna(x) else x
                )
        else:
            raise ValueError(f"Unsupported missing value strategy: '{strategy}' for column '{col}'")
        
    return X
missing_handler = FunctionTransformer(handle_missing_values, feature_names_out='one-to-one')



# Bound outliers 1.5x outside IQR
def handle_outliers(X):
    X = X.copy()
    for col in X.select_dtypes(include='number').columns:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        X[col] = X[col].clip(lower, upper)
    return X
outlier_handler = FunctionTransformer(handle_outliers, feature_names_out='one-to-one')



cleaning_pipeline = Pipeline([
    ('drop_columns', dropper),
    ('map_values', mapper),
    ('coerce_types', coercer),
    ('handle_missing', missing_handler),
    ('handle_outliers', outlier_handler)
])

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_scale_cols),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
], remainder='passthrough')