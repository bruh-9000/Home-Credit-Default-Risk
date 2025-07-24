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

from shared.utils import (
    cleaning_pipeline,
    preprocessor
)



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