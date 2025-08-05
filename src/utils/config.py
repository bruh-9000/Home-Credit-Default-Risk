from pathlib import Path
import yaml

ROOT_PATH = Path(__file__).resolve().parents[2]
SAVE_PATH = ROOT_PATH / 'saved'
CONFIG_PATH = ROOT_PATH / 'config.yaml'

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)


label = config['general']['label']
ID_feature = config['general']['ID_feature']
primary_metric = config['general']['primary_metric']
threshold = config['general']['threshold']
cv_splits = config['general']['cv_splits']
resampling_type = config['general']['resampling_type']
resampling_strategy = config['general']['resampling_strategy']

keep_features = config['preprocessing']['keep_features']
value_mappings = config['preprocessing']['value_mappings']
money_cols = config['preprocessing']['money_cols']
missing_handling = config['preprocessing']['missing_handling']
comb_low_freq = config['preprocessing']['comb_low_freq']
clip_features = config['preprocessing']['clip_features']
type_coercion = config['preprocessing']['type_coercion']

numerical_scale_cols = config['encoding']['numerical_scale_cols']
onehot_cols = config['encoding']['onehot_cols']
freq_cols = config['encoding']['freq_cols']
target_cols = config['encoding']['target_cols']
ordinal_cols = config['encoding']['ordinal_cols']
binned_cols = config['encoding']['binned_cols']

model_configs = config.get('model_configs', {})
