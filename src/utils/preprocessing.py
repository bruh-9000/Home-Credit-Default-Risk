import warnings
import re
import numpy as np
import pandas as pd
from category_encoders import CountEncoder, HashingEncoder, TargetEncoder
from matplotlib import pyplot as plt
import missingno as msno
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

warnings.filterwarnings('ignore')
np.random.seed(42)

from src.utils.config import (
    label,
    cv_splits,
    drop_or_keep,
    drop_features,
    keep_features,
    value_mappings,
    type_coercion,
    missing_handling,
    money_cols,
    numerical_scale_cols,
    onehot_cols,
    freq_cols,
    target_cols,
    ordinal_cols,
    binned_cols,
    hash_cols
)



def prepare_train_test_split(train_df):
    X = train_df
    y = train_df[label]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X, y, X_train, X_test, y_train, y_test



def dedup(df):
    num_dupes = df.duplicated().sum()
    if num_dupes > 0:
        print(f"Successfully deleted {num_dupes} duplicated examples.")
        df = df.drop_duplicates()
    return df



def handle_columns(X):
    assert label not in keep_features, f"Label '{label}' should not be in keep_features'"

    if drop_or_keep == 'keep':
        to_keep = [col for col in keep_features if col in X.columns]
        X = X[to_keep].copy()
        X = X.reset_index(drop=True)
    elif drop_or_keep == 'drop':
        to_drop = [col for col in drop_features if col in X.columns]
        X = X.drop(columns=to_drop)
        X = X.reset_index(drop=True)

    return pd.DataFrame(X)
column_handler = FunctionTransformer(handle_columns, validate=False, feature_names_out='one-to-one')



# Map values for features (e.g., 'yes' -> 1, 'no' -> 0)
def apply_mappings(X):
    for col, mapping in value_mappings.items():
        if col != label and col in X.columns:
            fixed_mapping = {np.nan if str(k).lower() == 'nan' else k: v for k, v in mapping.items()}
            X[col] = X[col].map(fixed_mapping)

            # Now explicitly fillna if mapping contained NaN
            if any(str(k).lower() == 'nan' for k in mapping.keys()):
                X[col] = X[col].fillna(mapping['NaN'])  # YAML str 'NaN'
    return X
mapper = FunctionTransformer(apply_mappings, feature_names_out='one-to-one')



def coerce_types(X):
    str_to_dtype = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool
    }

    type_coercion_actual = {k: str_to_dtype[v] for k, v in type_coercion.items()}

    return X.astype(type_coercion_actual)
coercer = FunctionTransformer(coerce_types, feature_names_out='one-to-one')



# Handle missing values in different ways
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
        elif strategy == 'prior': # Like dummy classifier, imputes missing with relative value probabilities. Good for cateogires
            non_na = X[col].dropna()
            if non_na.nunique() == 2:
                probs = non_na.value_counts(normalize=True)
                X[col] = X[col].apply(
                    lambda x: np.random.choice(probs.index, p=probs.values) if pd.isna(x) else x
                )
        else:
            X[col] = X[col].fillna(strategy) # Impute missing data with a set value
        
    return X
missing_handler = FunctionTransformer(handle_missing_values, feature_names_out='one-to-one')



# Convert any string monetary values into floats (e.g., '$1.23' into 1.23)
def convert_dollar_strings(X):
    X = X.copy()
    for col in money_cols:
        if col in X.columns:
            X[col] = X[col].apply(lambda x: re.sub(r'[\$,]', '', str(x)).strip() if pd.notna(x) else x)
            X[col] = X[col].replace('', np.nan)
            X[col] = pd.to_numeric(X[col], errors='coerce')
    return X
dollar_string_converter = FunctionTransformer(convert_dollar_strings, feature_names_out='one-to-one')



class NamedFunctionTransformer(FunctionTransformer):
    def __init__(self, func, feature_names):
        super().__init__(func)
        self.feature_names = feature_names

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names)



def hash_encode(X):
    return HashingEncoder(n_components=4).fit_transform(X)
hash_transformer = NamedFunctionTransformer(
    hash_encode, feature_names=hash_cols
)



cleaning_pipeline = Pipeline([
    ('handle_columns', column_handler),
    ('map_values', mapper),
    ('handle_missing', missing_handler),
    ('money_convert', dollar_string_converter),
    ('coerce_types', coercer)
])



preprocessor = ColumnTransformer([
    ('nums', StandardScaler(), numerical_scale_cols),
    ('oneh', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
    ('freq', CountEncoder(normalize=True), freq_cols),
    ('targ', TargetEncoder(cv=cv_splits), target_cols),
    ('ordi', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
    ('bins', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile'), binned_cols),
    ('hash', hash_transformer, hash_cols)
], remainder='passthrough')
preprocessor.set_output(transform='pandas')


def build_preprocessor(feature_columns):
    """Create a ColumnTransformer using only columns that exist in the data."""

    def filter_cols(cols):
        return [c for c in cols if c in feature_columns]

    proc = ColumnTransformer([
        ('nums', StandardScaler(), filter_cols(numerical_scale_cols)),
        ('oneh', OneHotEncoder(handle_unknown='ignore', sparse_output=False), filter_cols(onehot_cols)),
        ('freq', CountEncoder(normalize=True), filter_cols(freq_cols)),
        ('targ', TargetEncoder(cv=cv_splits), filter_cols(target_cols)),
        ('ordi', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), filter_cols(ordinal_cols)),
        ('bins', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile'), filter_cols(binned_cols)),
        ('hash', hash_transformer, filter_cols(hash_cols))
    ], remainder='passthrough')
    proc.set_output(transform='pandas')

    return proc



def show_categorical_uniques(df, limit=10):
    for col in df.select_dtypes(include=['object', 'category']).columns:
        uniques = df[col].unique()
        print(f"{col} ({len(uniques)} unique): {uniques[:limit]}")



def show_missing_data(df):
    if df.isna().sum().sum() > 0:
        missing_counts = df.isna().sum()
        print('Missing values detected in:')
        print(missing_counts[missing_counts > 0])

        # Limit to just columns with missing values
        missing_cols = df.loc[:, df.isna().any()]

        # Adjust figure size based on number of columns
        num_cols = missing_cols.shape[1]
        width = min(8, num_cols * 1.2)  # max width 8, scale with # columns
        height = 4

        plt.figure(figsize=(width, height))
        msno.matrix(missing_cols, figsize=(width, height), fontsize=10)
        plt.title('Missing Data by Column')
        plt.tight_layout()
        plt.show()
    else:
        print('No missing data found!')