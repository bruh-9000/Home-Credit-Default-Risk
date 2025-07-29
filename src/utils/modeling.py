import warnings
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
np.random.seed(42)

from src.utils.config import (
    label,
    primary_metric,
    threshold,
    cv_splits,
    resampling_type,
    resampling_strategy,
    model_configs
)

from src.utils.preprocessing import (
    cleaning_pipeline,
    preprocessor,
    build_preprocessor
)

skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)



models = {
    'dummy_classifier': DummyClassifier(strategy='stratified', random_state=42),
    'logistic_regression': LogisticRegression(random_state=42),
    'random_forest': RandomForestClassifier(random_state=42),
    'lightgbm': LGBMClassifier(verbose=-1, verbosity=-1, random_state=42)
}



def predict_pipeline(pipe, X_test):
    if hasattr(pipe.named_steps['model'], 'predict_proba'):
        y_probs = pipe.predict_proba(X_test)[:, 1]
        y_preds = (y_probs >= threshold).astype(int)
    else:
        y_probs = None
        y_preds = pipe.predict(X_test)

    return y_preds, y_probs



def prepare_data(X_train, X_test=None, y_train=None):
    cpipeline_clone = clone(cleaning_pipeline)
    preprocessor_clone = clone(preprocessor)

    cpipeline_clone.fit(X_train)
    X_train_clean = cpipeline_clone.transform(X_train)
    X_test_clean = cpipeline_clone.transform(X_test) if X_test is not None else None

    X_train_clean, y_train = X_train_clean.reset_index(drop=True), y_train.reset_index(drop=True)

    # Fit preprocessor
    if y_train is not None:
        preprocessor_clone.fit(X_train_clean, y_train)
    else:
        preprocessor_clone.fit(X_train_clean)

    # Transform both sets
    X_train_processed = preprocessor_clone.transform(X_train_clean)
    X_test_processed = preprocessor_clone.transform(X_test_clean) if X_test_clean is not None else None

    return X_train_processed, X_test_processed



def train_pipeline(name, X_train, y_train):
    config = model_configs.get(name, {})
    search_type = config.get('search_type')
    param_grid = config.get('param_grid', {})
    n_iter = config.get('n_iter', 10)

    def resample_and_restore(X, y, sampler):
        X_res, y_res = sampler.fit_resample(X, y)
        X_res = pd.DataFrame(X_res, columns=X.columns).reset_index(drop=True)
        y_res = pd.Series(y_res, name=y.name).reset_index(drop=True)
        
        return X_res, y_res

    if resampling_type == 'over':
        sampler = SMOTEENN(sampling_strategy=resampling_strategy, random_state=42)
        X_train, y_train = resample_and_restore(X_train, y_train, sampler)
    elif resampling_type == 'under':
        sampler = RandomUnderSampler(sampling_strategy=resampling_strategy, random_state=42)
        X_train, y_train = resample_and_restore(X_train, y_train, sampler)

    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)

    dynamic_preprocessor = build_preprocessor(X_train.columns)

    pipe = Pipeline([
        ('cleaning', cleaning_pipeline),
        ('encoding', dynamic_preprocessor),
        ('model', models[name])
    ])


    if search_type in ['grid', 'random']:
        if search_type == 'grid':
            search = GridSearchCV(pipe, param_grid=param_grid, cv=skf, scoring=primary_metric, n_jobs=-1)

        elif search_type == 'random':
            search = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=n_iter, cv=skf, scoring=primary_metric, n_jobs=-1)
        
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"\nBest hyperparameters:")
        print(search.best_params_)

    else:
        pipe.fit(X_train, y_train)
        best_model = pipe

    return best_model