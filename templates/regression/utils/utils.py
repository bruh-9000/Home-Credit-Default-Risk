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



# Data preparation utilities



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
        X_train, X_test, y_train, y_true = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_val = y_val = None

    else:
        X_test = test_df
        X_train = train_df
        y_train = y

        # Case 2: If test has labels, use as test set
        if label in test_df.columns:
            y_true = test_df[label]
            X_val = y_val = None

        # Case 3: If test has NO labels, make val set from train
        else:
            y_true = None

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
            )

    return X, y, X_train, X_test, y_train, y_true, X_val, y_val



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



# Model training utilities



models = {
    'dummy_regressor': DummyRegressor(strategy='mean', random_state=42),
    'linear_regression': LinearRegression(),
    'random_forest': RandomForestRegressor(random_state=42),
    'lightgbm': LGBMRegressor(verbose=-1, verbosity=-1, random_state=42)
}



def train_pipeline(name, X_train, y_train):
    config = model_configs.get(name, {})
    search_type = config.get('search_type')
    param_grid = config.get('param_grid', {})
    n_iter = config.get('n_iter', 10)

    skf = KFold(n_splits=5, shuffle=True, random_state=42)

    pipe = Pipeline([
        ('cleaning', cleaning_pipeline),
        ('preprocessing', preprocessor),
        ('model', models[name])
    ])

    if search_type == 'grid':
        search = GridSearchCV(pipe, param_grid=param_grid, cv=skf, scoring=primary_metric, n_jobs=-1)
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        print(f'\nBest hyperparameters for {name} (GridSearchCV):')
        print(search.best_params_)

    elif search_type == 'random':
        search = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=n_iter, cv=skf,
                                    scoring=primary_metric, n_jobs=-1)
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        print(f'\nBest hyperparameters for {name} (RandomizedSearchCV):')
        print(search.best_params_)

    else:
        scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring=primary_metric, n_jobs=-1)
        print(f'\n{name} - CV {primary_metric}: {scores.mean():.4f} ± {scores.std():.4f}')
        pipe.fit(X_train, y_train)

    return pipe



def predict_pipeline(pipe, X_test):
    y_preds = pipe.predict(X_test)
    return y_preds



# Evaluation and analysis utilities



def evaluate_pipeline(name, data):
    X_full, X_train, y_train, X_test, y_true, X_val, y_val = data

    # Get label mapping from config
    label_mapping = value_mappings.get(label)

    # Apply label mapping to y_train/y_true/y_val
    if label_mapping:
        y_train = y_train.map(label_mapping)
        if y_true is not None:
            y_true = y_true.map(label_mapping)
        if y_val is not None:
            y_val = y_val.map(label_mapping)

    # Check whether we have test labels
    has_test_labels = y_true is not None and not pd.isnull(y_true).all()

    # Train the pipeline
    pipe = train_pipeline(name, X_train, y_train)
    model = pipe.named_steps['model']

    # Train metrics
    y_train_preds = pipe.predict(X_train)
    print(f'\n{name} - TRAIN METRICS:')
    train_metrics = get_analysis(y_train, y_train_preds)

    y_preds = y_probs = None

    # Use test set if labels exist
    if has_test_labels:
        y_preds = predict_pipeline(pipe, X_test)
        print(f'\n{name} - TEST METRICS:')
        test_metrics = get_analysis(y_true, y_preds)
        output_metrics = test_metrics

        if not model_configs.get(name, {}).get('baseline', False):
            generate_graphs(pipe, model, X_full, y_true, y_preds, name)

    # Otherwise fall back to validation set
    elif y_val is not None:
        y_preds = predict_pipeline(pipe, X_val)
        print(f'\n{name} - VALIDATION METRICS:')
        val_metrics = get_analysis(y_val, y_preds)
        output_metrics = val_metrics

        if not model_configs.get(name, {}).get('baseline', False):
            generate_graphs(pipe, model, X_full, y_val, y_preds, name)

    else:
        print(f'\n{name} - SKIPPING TEST & VALIDATION (no labels)')

    joblib.dump(pipe, SAVE_DIR / f"{name}_pipeline.pkl")
    return pipe, output_metrics



def get_analysis(y_true, y_preds):
    mae = mean_absolute_error(y_true, y_preds)
    mse = mean_squared_error(y_true, y_preds)
    rmse = root_mean_squared_error(y_true, y_preds)
    r2 = r2_score(y_true, y_preds)

    report = f'''
MAE: {mae:.2%}
MSE: {mse:.2%}
RMSE: {rmse:.2%}
R2: {r2:.2%}
'''
    
    values = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

    print(report)

    return values



def generate_graphs(pipeline, model, X, y_true, y_pred, name):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_shap_summary(pipeline, model, X)



def plot_shap_summary(pipeline, model, X, num_samples=100):
    X_cleaned = pipeline.named_steps['cleaning'].transform(X)
    X_transformed = pipeline.named_steps['preprocessing'].transform(X_cleaned)
    features = pipeline.named_steps['preprocessing'].get_feature_names_out()
    X_sample = pd.DataFrame(X_transformed[:num_samples], columns=features)

    explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap.summary_plot(shap_values, X_sample, feature_names=features, plot_type='bar', plot_size=[8,4])



def summarize_model_results(model_data, primary_metric, metrics_to_display, pipelines=None):
    df = pd.DataFrame(model_data).T
    primary_metric = primary_metric

    df_sorted = df.sort_values(by=primary_metric, ascending=False)

    worst_model = df_sorted.index[-1]
    best_model = df_sorted.index[0]

    styled = df_sorted[list(metrics_to_display.keys())].style.format(metrics_to_display)

    display(Markdown(f'**Worst model by {primary_metric}:** {worst_model} with a score of {df_sorted.loc[worst_model, primary_metric]:.2%}'))
    display(Markdown(f'**Best model by {primary_metric}:** {best_model} with a score of {df_sorted.loc[best_model, primary_metric]:.2%}'))
    display(styled)

    return best_model, pipelines[best_model]