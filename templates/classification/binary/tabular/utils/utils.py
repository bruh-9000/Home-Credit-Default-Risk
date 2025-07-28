import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(42)
import joblib
import yaml
import re
from ruamel.yaml import YAML
import pandas as pd
from IPython.display import display, Markdown
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    KBinsDiscretizer,
    FunctionTransformer,
)
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.hashing import HashingEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SAVE_DIR = ROOT / "saved"

label: str
primary_metric: str
drop_features: list
value_mappings: dict
type_coercion: dict
missing_handling: dict
money_cols: list
numerical_scale_cols: list
onehot_cols: list
freq_cols: list
ordinal_cols: list
binned_cols: list
hash_cols: list
model_configs: dict

def load_config():
    with open(ROOT / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    globals().update({
        "config": config,
        "label": config["general"]["label"],
        "primary_metric": config["general"]["primary_metric"],
        "drop_features": config["preprocessing"]["drop_features"],
        "value_mappings": config["preprocessing"]["value_mappings"],
        "type_coercion": config["preprocessing"]["type_coercion"],
        "missing_handling": config["preprocessing"]["missing_handling"],
        "money_cols": config["preprocessing"]["money_cols"],
        "numerical_scale_cols": config["encoding"]["numerical_scale_cols"],
        "onehot_cols": config["encoding"]["onehot_cols"],
        "freq_cols": config["encoding"]["freq_cols"],
        "ordinal_cols": config["encoding"]["ordinal_cols"],
        "binned_cols": config["encoding"]["binned_cols"],
        "hash_cols": config["encoding"]["hash_cols"],
        "model_configs": config.get("model_configs", {})
    })

load_config()

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
import shap
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno



def load_data(file_one, file_two=None):
    df1 = pd.read_csv(file_one)
    df2 = pd.read_csv(file_two) if file_two else None
    return df1, df2



def dedup(df):
    num_dupes = df.duplicated().sum()
    if num_dupes > 0:
        print(f"Successfully deleted {num_dupes} duplicated examples.")
        df = df.drop_duplicates()
    return df



def prepare_train_test_split(train_df, test_df):
    X = train_df
    y = train_df[label]

    if test_df is None:
        # Case 1: Only one dataset, do train/test split, then val from train
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
        )

    else:
        X_test = test_df.copy()

        if label in X_test.columns:
            # Case 2: Test set includes labels
            y_test = X_test[label]
            X_test = X_test.drop(columns=[label])
        else:
            # Case 3: Test set is unlabeled
            y_test = None

        # Split train into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

    return X, y, X_train, X_test, y_train, y_test, X_val, y_val



# Data preparation utilities



def auto_config_from_data(X, CONFIG_PATH):
    yaml = YAML()
    yaml.preserve_quotes = True

    with open(CONFIG_PATH, "r") as f:
        config = yaml.load(f)

    drop_features = set()
    value_mappings = {}
    money_cols = set()
    type_coercion = {}
    missing_handling = {}

    # Drop: 90%+ missing or constant
    high_missing = X.columns[X.isna().mean() > 0.9]
    constant_cols = X.columns[X.nunique(dropna=False) <= 1]
    drop_features |= set(high_missing) | set(constant_cols)
    config["preprocessing"]["drop_features"] = sorted(drop_features)

    # Binary string value mappings
    for col in X.columns:
        if col in drop_features:
            continue
        uniques = X[col].unique()
        if len(uniques) == 2 and all(isinstance(v, str) for v in uniques):
            sorted_vals = sorted(uniques)
            value_mappings[col] = {sorted_vals[0]: 0, sorted_vals[1]: 1}
    config["preprocessing"]["value_mappings"] = value_mappings

    # Money-like columns
    money_regex = re.compile(r"^\$\d{1,3}(,\d{3})*(\.\d{2})?$|^\$\d+(\.\d{2})?$")
    for col in X.columns:
        if col in drop_features:
            continue
        sample = X[col].dropna().astype(str).head(10)
        if sample.str.contains(r"\$").any():
            match_rate = sample.apply(lambda x: bool(money_regex.match(x.strip()))).mean()
            if match_rate >= 0.8:
                money_cols.add(col)
    config["preprocessing"]["money_cols"] = sorted(money_cols)

    # Type coercion (str that can be floats)
    for col in X.select_dtypes(include="object"):
        if col in drop_features or col == label:
            continue
        try:
            X[col].astype(float)
            type_coercion[col] = "float"
        except Exception:
            pass
    config["preprocessing"]["type_coercion"] = dict(sorted(type_coercion.items()))

    # Missing handling
    for col in X.columns:
        if col in drop_features or col == label:
            continue
        ratio = X[col].isna().mean()
        if ratio == 0 or ratio > 0.5:
            continue
        nunique = X[col].nunique(dropna=True)
        if np.issubdtype(X[col].dtype, np.number):
            skew = X[col].skew(skipna=True)
            if nunique <= 3:
                missing_handling[col] = "mode"
            elif abs(skew) < 1:
                missing_handling[col] = "mean"
            else:
                missing_handling[col] = "median"
        elif X[col].dtype == object and nunique <= 10:
            missing_handling[col] = "mode"
    config["preprocessing"]["missing_handling"] = dict(sorted(missing_handling.items()))

    # Write out
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

    load_config()

    print(f"Updated drop_features: {sorted(drop_features)}")
    print(f"Updated value_mappings: {sorted(value_mappings)}")
    print(f"Updated money_cols: {sorted(money_cols)}")
    print(f"Updated type_coercion: {sorted(type_coercion)}")
    print(f"Updated missing_handling: {sorted(missing_handling)}")



def show_categorical_uniques(df, limit=10):
    for col in df.select_dtypes(include=["object", "category"]).columns:
        uniques = df[col].unique()
        print(f"{col} ({len(uniques)} unique): {uniques[:limit]}")



def show_missing_data(df):
    # Check missing data percentages per column
    if df.isna().sum().sum() > 0:
        missing_counts = df.isna().sum()

        # Print columns with missing data
        print('Missing values detected in:')
        print(missing_counts[missing_counts > 0])

        # Plot
        msno.matrix(df, figsize=(8.5, 4), fontsize=10)
        plt.title('Missing Data by Column')
        plt.show()
    else:
        print("No missing data found!")



# Feature engineering pipeline



def drop_columns(X):
    if isinstance(X, pd.DataFrame):
        to_drop = [col for col in drop_features if col in X.columns]
        X = X.drop(columns=to_drop)
        X = X.reset_index(drop=True)
    return pd.DataFrame(X)
dropper = FunctionTransformer(drop_columns, validate=False)



# Map values for features (e.g., 'yes' -> 1, 'no' -> 0)
def apply_mappings(X):
    for col, mapping in value_mappings.items():
        if col != label and col in X.columns:
            fixed_mapping = {np.nan if str(k).lower() == 'nan' else k: v for k, v in mapping.items()}
            X[col] = X[col].map(fixed_mapping)

            # Now explicitly fillna if mapping contained NaN
            if any(str(k).lower() == 'nan' for k in mapping.keys()):
                X[col] = X[col].fillna(mapping['NaN'])  # YAML str "NaN"
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



def freq_encode(X):
    X = X.copy()
    for col in X.columns:
        freqs = X[col].value_counts(normalize=True)
        X[col] = X[col].map(freqs).fillna(0)
    return X
freq_transformer = NamedFunctionTransformer(
    freq_encode, feature_names=freq_cols
)



def hash_encode(X):
    return HashingEncoder(n_components=4).fit_transform(X)
hash_transformer = NamedFunctionTransformer(
    hash_encode, feature_names=hash_cols
)
    


cleaning_pipeline = Pipeline([
    ('drop_columns', dropper),
    ('map_values', mapper),
    ('handle_missing', missing_handler),
    ('money_convert', dollar_string_converter),
    ('coerce_types', coercer)
])

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_scale_cols),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
    ('freq', freq_transformer, freq_cols),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
    ('binned', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile'), binned_cols),
    ('hashed', hash_transformer, hash_cols)
], remainder='passthrough')
preprocessor.set_output(transform='pandas')



# Pipeline, evaluation, and analysis



models = {
    'dummy_classifier': DummyClassifier(strategy='stratified', random_state=42),
    'logistic_regression': LogisticRegression(random_state=42),
    'random_forest': RandomForestClassifier(random_state=42),
    'lightgbm': LGBMClassifier(verbose=-1, verbosity=-1, random_state=42)
}



def predict_pipeline(pipe, X_test):
    y_preds = pipe.predict(X_test)
    y_probs = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps['model'], "predict_proba") else None
    return y_preds, y_probs



def evaluate_pipeline(name, data):
    X_full, X_train, y_train, X_test, y_test, X_val, y_val = data

    # Get label mapping from config
    label_mapping = value_mappings.get(label)

    # Apply label mapping to y_train/y_test/y_val
    if label_mapping:
        y_train = y_train.map(label_mapping)
        if y_test is not None:
            y_test = y_test.map(label_mapping)
        if y_val is not None:
            y_val = y_val.map(label_mapping)

    # Check whether there are test labels
    has_test_labels = y_test is not None and not pd.isna(y_test).all()

    # Train the pipeline
    pipe = train_pipeline(name, X_train, y_train)
    model = pipe.named_steps['model']

    # Train metrics
    y_train_preds = pipe.predict(X_train)
    print(f'\n{name} - TRAIN METRICS:')
    train_metrics = get_analysis(y_train, y_train_preds) # Var not used, but func prints stuff

    y_preds = y_probs = None

    # Use test set if labels exist
    if has_test_labels:
        y_preds, y_probs = predict_pipeline(pipe, X_test)
        print(f'\n{name} - TEST METRICS:')
        test_metrics = get_analysis(y_test, y_preds)
        output_metrics = test_metrics

        if not model_configs.get(name, {}).get('baseline', False):
            generate_graphs(model, X_train, y_test, y_preds, y_probs)

    # Otherwise fall back to validation set
    elif y_val is not None:
        y_preds, y_probs = predict_pipeline(pipe, X_val)
        print(f'\n{name} - VALIDATION METRICS:')
        val_metrics = get_analysis(y_val, y_preds)
        output_metrics = val_metrics

        if not model_configs.get(name, {}).get('baseline', False):
            generate_graphs(model, X_train, y_val, y_preds, y_probs)

    else:
        print(f'\n{name} - SKIPPING TEST & VALIDATION (no labels)')

    joblib.dump(pipe, SAVE_DIR / f"{name}_pipeline.pkl")
    return pipe, output_metrics



def train_pipeline(name, X_train, y_train):
    config = model_configs.get(name, {})
    search_type = config.get('search_type')
    param_grid = config.get('param_grid', {})
    n_iter = config.get('n_iter', 10)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipe = Pipeline([
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



def get_analysis(y_test, y_preds):
    tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()
    accuracy = accuracy_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds)
    recall = recall_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds)
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    report = f'''
Accuracy: {accuracy:.2%}
F1 Score: {f1:.2%}
Precision: {precision:.2%}
True Positive Rate (Recall): {recall:.2%}
True Negative Rate (Specificity): {specificity:.2%}
False Positive Rate: {fpr:.2%}
False Negative Rate: {fnr:.2%}
'''
    
    values = {
        'Accuracy': accuracy,
        'F1': f1,
        'Precision': precision,
        'Recall (TPR)': recall,
        'Specificity (TNR)': specificity,
        'FPR': fpr,
        'FNR': fnr
    }

    print(report)

    return values



def generate_graphs(model, X_train, y_test, y_pred, y_probs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_confusion_matrix(y_test, y_pred, ax=axes[0])
    plot_roc_curve(y_test, y_probs, ax=axes[1])
    plot_pr_curve(y_test, y_probs, ax=axes[2])

    plt.tight_layout()
    plt.show()

    plot_shap_summary(model, X_train)



def plot_confusion_matrix(y_test, y_pred, labels=['Negative', 'Positive'], ax=None):
    cm = confusion_matrix(y_test, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)



def plot_roc_curve(y_test, y_probs, ax=None):
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    ax.grid(True)



def plot_pr_curve(y_test, y_probs, ax=None):
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    avg_precision = average_precision_score(y_test, y_probs)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, label=f'Avg Precision = {avg_precision:.4f}')
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(True)



def plot_shap_summary(model, X_train, num_samples=20):
    features = X_train.columns
    X_sample = pd.DataFrame(X_train[:num_samples], columns=features)

    explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap.summary_plot(shap_values, X_sample, feature_names=features, plot_type='bar', plot_size=[8, 4])



def summarize_model_results(model_data, primary_metric, metrics_to_display, pipelines=None):
    df = pd.DataFrame(model_data).T
    primary_metric = primary_metric.capitalize()

    df_sorted = df.sort_values(by=primary_metric, ascending=False)

    worst_model = df_sorted.index[-1]
    best_model = df_sorted.index[0]

    styled = df_sorted[list(metrics_to_display.keys())].style.format(metrics_to_display)

    display(Markdown(f'**Worst model by {primary_metric}:** {worst_model} with a score of {df_sorted.loc[worst_model, primary_metric]:.2%}'))
    display(Markdown(f'**Best model by {primary_metric}:** {best_model} with a score of {df_sorted.loc[best_model, primary_metric]:.2%}'))
    display(styled)

    return best_model, pipelines[best_model]