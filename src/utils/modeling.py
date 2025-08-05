from src.utils.preprocessing import (
    cleaning_pipeline,
    cleaning_list,
    preprocessor,
    post_encoding_sanitizer
)
from src.utils.config import (
    primary_metric,
    threshold,
    cv_splits,
    resampling_type,
    resampling_strategy,
    model_configs
)
import warnings
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
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')


models = {
    'dummy_classifier': DummyClassifier(strategy='stratified', random_state=42),
    'logistic_regression': LogisticRegression(random_state=42),
    'random_forest': RandomForestClassifier(random_state=42),
    'lightgbm': LGBMClassifier(verbose=-1, verbosity=-1, random_state=42)
}


def predict_pipeline(pipe, X_val):
    if hasattr(pipe.named_steps['model'], 'predict_proba'):
        y_probs = pipe.predict_proba(X_val)[:, 1]
        y_preds = (y_probs >= threshold).astype(int)
    else:
        y_probs = None
        y_preds = pipe.predict(X_val)

    return y_preds, y_probs


def prepare_data(X_train, X_val=None, y_train=None):
    ccleaner = clone(cleaning_pipeline)
    cpreprocessor = clone(preprocessor)

    ccleaner.fit(X_train)
    X_train_clean = ccleaner.transform(X_train)

    X_val_clean = None
    if X_val is not None:
        X_val_clean = ccleaner.transform(X_val)

    if y_train is not None:
        cpreprocessor.fit(X_train_clean, y_train)
    else:
        cpreprocessor.fit(X_train_clean)

    X_train_processed = cpreprocessor.transform(X_train_clean)
    X_train_processed = post_encoding_sanitizer.transform(X_train_processed)

    X_val_processed = None
    if X_val_clean is not None:
        X_val_processed = cpreprocessor.transform(X_val_clean)
        X_val_processed = post_encoding_sanitizer.transform(X_val_processed)

    return X_train_processed, X_val_processed


# For 01_eda.ipynb, keeps missing data
def prepare_eda_data(X_train):
    cpipeline = Pipeline([
        step for step in cleaning_list
        if step[0] not in ('handle_missing', 'coerce_types')
    ])

    cpipeline.fit(X_train)
    X_train_clean = cpipeline.transform(X_train)
    return X_train_clean


def train_pipeline(name, X_train, y_train):
    config = model_configs.get(name, {})
    search_type = config.get('search_type')
    param_grid = config.get('param_grid', {})
    n_iter = config.get('n_iter', 10)

    cpreprocessor = clone(preprocessor)

    steps = [
        *cleaning_list,
        ('encoding', cpreprocessor),
        ('sanitize_encoded', post_encoding_sanitizer),
    ]

    # Add sampling if set in config
    if resampling_type == 'over':
        sampler = SMOTEENN(sampling_strategy=resampling_strategy, random_state=42)
        steps.append(('resample', sampler))
    elif resampling_type == 'under':
        sampler = RandomUnderSampler(sampling_strategy=resampling_strategy, random_state=42)
        steps.append(('resample', sampler))

    steps.append(('model', models[name]))

    pipe = IMBPipeline(steps)

    if search_type in ['grid', 'random']:
        scoring = {
            'accuracy': 'accuracy',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall',
            'roc_auc': 'roc_auc',
        }

        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

        if search_type == 'grid':
            search = GridSearchCV(pipe, param_grid=param_grid, cv=skf,
                                  scoring=scoring, refit=primary_metric, n_jobs=-1)

        elif search_type == 'random':
            search = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=n_iter,
                                        cv=skf, scoring=scoring, refit=primary_metric, n_jobs=-1,
                                        random_state=42)

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print("\nBest hyperparameters:")
        print(search.best_params_)

    else:
        pipe.fit(X_train, y_train)
        best_model = pipe

    if search_type in ['grid', 'random']:
        return best_model, search
    else:
        return best_model, None
