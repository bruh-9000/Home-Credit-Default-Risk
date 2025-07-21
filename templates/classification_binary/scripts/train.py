import argparse
from pathlib import Path
import sys
import os
import yaml
import joblib

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT / "saved"
CONFIG_PATH = ROOT / "config.yaml"
sys.path.append(str(ROOT))

# CLI parser
parser = argparse.ArgumentParser(description="Train a pipeline")
parser.add_argument("model_name", type=str, help="Model to train (e.g., lightgbm, random_forest)")
args = parser.parse_args()
model_name = args.model_name

# Load config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

primary_metric = config["general"]["primary_metric"]

if not (SAVE_DIR / "X_train.pkl").exists():
    raise FileNotFoundError("Missing 'saved/X_train.pkl'. Run the preprocessing notebook first.")

# Load data
X_train = joblib.load(SAVE_DIR / "X_train.pkl")
y_train = joblib.load(SAVE_DIR / "y_train.pkl")

# Import and train
from utils.utils import train_and_predict_pipeline
pipe, _, _ = train_and_predict_pipeline(model_name, X_train, y_train, X_train, y_train)

# Save
joblib.dump(pipe, SAVE_DIR / f"{model_name}_pipeline.pkl")
print(f"Model saved to {SAVE_DIR / f'{model_name}_pipeline.pkl'}")