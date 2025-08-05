from src.utils.config import (
    label,
    cv_splits,
    keep_features,
    value_mappings,
    type_coercion,
    missing_handling,
    money_cols,
    comb_low_freq,
    clip_features,
    numerical_scale_cols,
    onehot_cols,
    freq_cols,
    target_cols,
    ordinal_cols,
    binned_cols
)
import re
import numpy as np
import pandas as pd
from category_encoders import CountEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder
)
from sklearn.model_selection import train_test_split

np.random.seed(42)


def prepare_train_val_split(train_df):
    X = train_df
    y = train_df[label]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X, y, X_train, X_val, y_train, y_val


def dedup(df):
    num_dupes = df.duplicated().sum()
    if num_dupes > 0:
        print(f"Successfully deleted {num_dupes} duplicated examples.")
        df = df.drop_duplicates()
    else:
        print("No duplicated examples found!")
    return df


def handle_columns(X):
    to_keep = [col for col in keep_features if col in X.columns]
    X = X[to_keep].copy()

    return X


column_handler = FunctionTransformer(
    handle_columns, validate=False, feature_names_out='one-to-one')


# Map values for features (e.g., 'yes' -> 1, 'no' -> 0)
def apply_mappings(X):
    for col, mapping in value_mappings.items():
        if col != label and col in X.columns:
            # Replace 'NaN' in config with np.nan
            clean_mapping = {
                k: (np.nan if str(v).lower() == 'nan' else v)
                for k, v in mapping.items()
            }
            X[col] = X[col].replace(clean_mapping)

    return X


mapper = FunctionTransformer(apply_mappings, validate=False, feature_names_out='one-to-one')


def coerce_types(X):
    str_to_dtype = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool
    }

    type_coerced = {k: str_to_dtype[v] for k, v in type_coercion.items()}

    return X.astype(type_coerced)


coercer = FunctionTransformer(coerce_types, validate=False, feature_names_out='one-to-one')


# Handle missing values in different ways
def handle_missing_values(X):
    if missing_handling is None:
        return X

    X = X.copy().replace('NaN', np.nan)

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
        elif strategy == 'prior':  # Like dummy classifier, imputate missing with relative probs
            non_na = X[col].dropna()
            if not non_na.empty:
                probs = non_na.value_counts(normalize=True)
                rng = np.random.default_rng(42)
                X[col] = X[col].apply(
                    lambda x: rng.choice(
                        probs.index, p=probs.values) if pd.isna(x) else x
                )
        else:
            # Impute missing data with a set value
            X[col] = X[col].fillna(strategy)

    return X


missing_handler = FunctionTransformer(
    handle_missing_values, validate=False, feature_names_out='one-to-one')


# Convert from money to int. Ex. '$1.00' to 1.00
def convert_dollar_strings(X):
    X = X.copy()
    for col in money_cols:
        if col in X.columns:
            X[col] = X[col].apply(lambda x: re.sub(
                r'[\$,]', '', str(x)).strip() if pd.notna(x) else x)
            X[col] = X[col].replace('', np.nan)
            X[col] = pd.to_numeric(X[col], errors='coerce')

    return X


dollar_string_converter = FunctionTransformer(
    convert_dollar_strings, validate=False, feature_names_out='one-to-one')


def clip_values(X):
    X = X.copy()

    if clip_features is None:
        return X

    for col, (low_q, high_q) in clip_features.items():
        if col in X.columns:
            lower = X[col].quantile(low_q / 100)
            upper = X[col].quantile(high_q / 100)
            X[col] = X[col].clip(lower=lower, upper=upper)

    return X


clipper = FunctionTransformer(clip_values, validate=False, feature_names_out='one-to-one')


def comb_cats(X):
    X = X.copy()

    for col, (threshold, replacement) in comb_low_freq.items():
        threshold /= 100
        freqs = X[col].value_counts(normalize=True)
        low_freq_vals = freqs[freqs <= threshold].index
        X[col] = X[col].apply(
            lambda val: replacement if val in low_freq_vals else val)

    return X


comb = FunctionTransformer(comb_cats, validate=False, feature_names_out='one-to-one')


class NamedFunctionTransformer(FunctionTransformer):
    def __init__(self, func, feature_names):
        super().__init__(func)
        self.feature_names = feature_names

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names)


# Remove any characters from feature names that might cause errors
def sanitize_after_encoding(X):
    X = X.copy()
    X.columns = [re.sub(r'[^\w]', '_', str(col)) for col in X.columns]

    return X


post_encoding_sanitizer = FunctionTransformer(
    sanitize_after_encoding, validate=False, feature_names_out='one-to-one')


cleaning_list = [
    ('handle_columns', column_handler),
    ('map_values', mapper),
    ('handle_missing', missing_handler),
    ('money_convert', dollar_string_converter),
    ('comb', comb),
    ('coerce_types', coercer),
    ('clipper', clipper)
]
cleaning_pipeline = Pipeline(cleaning_list)


preprocessor = ColumnTransformer([
    ('nums', StandardScaler(), numerical_scale_cols),
    ('oneh', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
    ('freq', CountEncoder(normalize=True), freq_cols),
    ('targ', TargetEncoder(cv=cv_splits, random_state=42), target_cols),
    ('ordi', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
    ('bins', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile'), binned_cols)
], remainder='passthrough')
preprocessor.set_output(transform='pandas')
