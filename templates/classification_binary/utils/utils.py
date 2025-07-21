# Core libraries
import warnings
import importlib

import pandas as pd
import utils.config as config
from IPython.display import display, Markdown
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from utils.config import (
    primary_metric,
    drop_features,
    value_mappings,
    type_coercion,
    missing_handling,
    numerical_cols,
    onehot_cols,
    ordinal_cols,
    model_configs,
)
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

importlib.reload(config)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data preparation utilities (used in `0_prep.ipynb`)
# ---------------------------------------------------------------------------

def load_data(file_one, file_two=None):
    """Load one or two CSV files into Pandas DataFrames."""
    if file_two is None:
        return pd.read_csv(file_one), None
    return pd.read_csv(file_one), pd.read_csv(file_two)


def dedup(df):
    """Drop duplicate rows and log how many were removed."""
    num_dupes = df.duplicated().sum()
    if num_dupes > 0:
        print(f"Successfully deleted {num_dupes} duplicated examples.")
        df = df.drop_duplicates()
    return df


def prepare_train_test_split(train_df, test_df, label):
    if test_df is None:
        X = train_df
        y = train_df[label]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        return X, y, X_train, X_test, y_train, y_test, None
    else:
        X_train = train_df
        y_train = train_df[label]
        X_test = test_df if label in test_df.columns else test_df
        y_test = test_df[label] if label in test_df.columns else None
        return X, y, X_train, X_test, y_train, y_test, None



# ---------------------------------------------------------------------------
# Feature engineering pipeline (used during training)
# ---------------------------------------------------------------------------

# Drop columns based on config.py list
def drop_columns(X):
    return X.drop(columns=drop_features)
dropper = FunctionTransformer(drop_columns, feature_names_out='one-to-one')



# Map values for features (e.g., 'yes' -> 1, 'no' -> 0)
def apply_mappings(X):
    X = X.copy()
    for col, mapping in value_mappings.items():
        X[col] = X[col].map(mapping)
    return X
mapper = FunctionTransformer(apply_mappings, feature_names_out='one-to-one')



# Make sure each feature is the correct type
def coerce_types(X):
    return X.astype(type_coercion)
coercer = FunctionTransformer(coerce_types, feature_names_out='one-to-one')



# Perform actions based on config.py setting for each feature
def handle_missing_values(X):
    X = X.copy()
    for col, strategy in missing_handling.items():
        if strategy == 'drop':
            X = X.dropna(subset=[col])
        else:
            imputer = SimpleImputer(strategy=strategy)
            X[[col]] = imputer.fit_transform(X[[col]])
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
    ('num', StandardScaler(), numerical_cols),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
], remainder='passthrough')

# ---------------------------------------------------------------------------
# Model training utilities
# ---------------------------------------------------------------------------

models = {
    'dummy_classifier': DummyClassifier(strategy='stratified', random_state=42),
    'logistic_regression': LogisticRegression(random_state=42),
    'random_forest': RandomForestClassifier(random_state=42),
    'lightgbm': LGBMClassifier(verbose=-1, verbosity=-1, random_state=42)
}



def train_and_predict_pipeline(name, X_train, y_train, X_test, y_test=None):
    config = model_configs.get(name, {})
    search_type = config.get('search_type')
    param_grid = config.get('param_grid', {})
    n_iter = config.get('n_iter', 10)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
        pipe.fit(X_train, y_train)

    y_preds = pipe.predict(X_test)
    y_probs = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps['model'], "predict_proba") else None

    return pipe, y_preds, y_probs


# ---------------------------------------------------------------------------
# Evaluation and analysis utilities
# ---------------------------------------------------------------------------

def evaluate_pipeline(name, data):
    X = data[0]
    X_train = data[1]
    y_train = data[2]
    X_test = data[3]
    y_test = data[4]

    pipe, y_preds, y_probs = train_and_predict_pipeline(name, X_train, y_train, X_test, y_test)

    y_train_preds = pipe.predict(X_train)
    model = pipe.named_steps['model']

    print(f'\n{name} - TRAIN METRICS:')
    train_metrics = get_analysis(y_train, y_train_preds)

    print(f'\n{name} - TEST METRICS:')
    test_metrics = get_analysis(y_test, y_preds)

    generate_graphs(pipe, model, X, y_test, y_preds, y_probs, name)

    return pipe, test_metrics



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



def generate_graphs(pipeline, model, X, y_test, y_pred, y_probs, name):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_confusion_matrix(y_test, y_pred, ax=axes[0])
    plot_roc_curve(y_test, y_probs, ax=axes[1])
    plot_pr_curve(y_test, y_probs, ax=axes[2])

    plt.tight_layout()
    plt.show()

    if not model_configs.get(name, {}).get('baseline', False):
        plot_shap_summary(pipeline, model, X)



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



def plot_shap_summary(pipeline, model, X, num_samples=100):
    X_transformed = pipeline.named_steps['preprocessing'].transform(X)
    features = pipeline.named_steps['preprocessing'].get_feature_names_out()
    X_sample = pd.DataFrame(X_transformed[:num_samples], columns=features)

    explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap.summary_plot(shap_values, X_sample, feature_names=features, plot_type='bar', plot_size=[8,4])



def summarize_model_results(model_data, primary_metric, metrics_to_display, pipelines=None):
    df = pd.DataFrame(model_data).T
    primary_metric = primary_metric.upper()
    df_sorted = df.sort_values(by=primary_metric, ascending=False)

    worst_model = df_sorted.index[-1]
    best_model = df_sorted.index[0]

    styled = df_sorted[list(metrics_to_display.keys())].style.format(metrics_to_display)

    display(Markdown(f'**Worst model by {primary_metric}:** {worst_model} with a score of {df_sorted.loc[worst_model, primary_metric]:.2%}'))
    display(Markdown(f'**Best model by {primary_metric}:** {best_model} with a score of {df_sorted.loc[best_model, primary_metric]:.2%}'))
    display(styled)

    if pipelines:
        return best_model, pipelines[best_model]
    else:
        return best_model