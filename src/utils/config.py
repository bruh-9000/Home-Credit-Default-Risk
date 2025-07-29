import warnings
import re
from pathlib import Path
import numpy as np
import yaml
from ruamel.yaml import YAML

warnings.filterwarnings('ignore')
np.random.seed(42)

ROOT_PATH = Path(__file__).resolve().parents[2]
SAVE_PATH = ROOT_PATH / 'saved'
CONFIG_PATH = ROOT_PATH / 'config.yaml'

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

def load_config():
    global label, primary_metric, threshold, cv_splits, resampling_type, resampling_strategy, drop_or_keep, drop_features, keep_features
    global value_mappings, type_coercion, missing_handling, money_cols
    global numerical_scale_cols, onehot_cols, freq_cols, target_cols, ordinal_cols, binned_cols, hash_cols
    global model_configs

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    label = config['general']['label']
    primary_metric = config['general']['primary_metric']
    threshold = config['general']['threshold']
    cv_splits = config['general']['cv_splits']
    resampling_type = config['general']['resampling_type']
    resampling_strategy = config['general']['resampling_strategy']
    drop_or_keep = config['general']['drop_or_keep']
    drop_features = config['preprocessing']['drop_features']
    keep_features = config['preprocessing']['keep_features']
    value_mappings = config['preprocessing']['value_mappings']
    type_coercion = config['preprocessing']['type_coercion']
    missing_handling = config['preprocessing']['missing_handling']
    money_cols = config['preprocessing']['money_cols']
    numerical_scale_cols = config['encoding']['numerical_scale_cols']
    onehot_cols = config['encoding']['onehot_cols']
    freq_cols = config['encoding']['freq_cols']
    target_cols = config['encoding']['target_cols']
    ordinal_cols = config['encoding']['ordinal_cols']
    binned_cols = config['encoding']['binned_cols']
    hash_cols = config['encoding']['hash_cols']
    model_configs = config.get('model_configs', {})

load_config()



def auto_config_from_data(X, CONFIG_PATH):
    yaml = YAML()
    yaml.preserve_quotes = True

    with open(CONFIG_PATH, 'r') as f:
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
    config['preprocessing']['drop_features'] = sorted(drop_features)

    # Binary string value mappings
    for col in X.columns:
        if col in drop_features:
            continue
        uniques = X[col].unique()
        if len(uniques) == 2 and all(isinstance(v, str) for v in uniques):
            sorted_vals = sorted(uniques)
            value_mappings[col] = {sorted_vals[0]: 0, sorted_vals[1]: 1}
    config['preprocessing']['value_mappings'] = value_mappings

    # Money-like columns
    money_regex = re.compile(r'^\$\d{1,3}(,\d{3})*(\.\d{2})?$|^\$\d+(\.\d{2})?$')
    for col in X.columns:
        if col in drop_features:
            continue
        sample = X[col].dropna().astype(str).head(10)
        if sample.str.contains(r'\$').any():
            match_rate = sample.apply(lambda x: bool(money_regex.match(x.strip()))).mean()
            if match_rate >= 0.8:
                money_cols.add(col)
    config['preprocessing']['money_cols'] = sorted(money_cols)

    # Type coercion (str that can be floats)
    for col in X.select_dtypes(include='object'):
        if col in drop_features or col == label:
            continue
        try:
            X[col].astype(float)
            type_coercion[col] = 'float'
        except Exception:
            pass
    config['preprocessing']['type_coercion'] = dict(sorted(type_coercion.items()))

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
                missing_handling[col] = 'mode'
            elif abs(skew) < 1:
                missing_handling[col] = 'mean'
            else:
                missing_handling[col] = 'median'
        elif X[col].dtype == object and nunique <= 10:
            missing_handling[col] = 'mode'
    config['preprocessing']['missing_handling'] = dict(sorted(missing_handling.items()))

    # Write out
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f)

    load_config()

    print(f'Updated drop_features: {sorted(drop_features)}')
    print(f'Updated value_mappings: {sorted(value_mappings)}')
    print(f'Updated money_cols: {sorted(money_cols)}')
    print(f'Updated type_coercion: {sorted(type_coercion)}')
    print(f'Updated missing_handling: {sorted(missing_handling)}')