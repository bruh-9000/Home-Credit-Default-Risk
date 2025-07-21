import argparse
from pathlib import Path
import sys
import joblib
import pandas as pd
import yaml

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT / "saved"
CONFIG_PATH = ROOT / "config.yaml"
sys.path.append(str(ROOT))

# CLI parser
parser = argparse.ArgumentParser(description="Make predictions with a trained model")
parser.add_argument("model_name", type=str, help="Model to use for prediction (e.g., lightgbm)")
args = parser.parse_args()
model_name = args.model_name

# Load config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

if not (SAVE_DIR / f"{model_name}_pipeline.pkl").exists():
    raise FileNotFoundError(f"Missing '{model_name}_pipeline.pkl'. Run the training first.")

# Load model and data
pipe = joblib.load(SAVE_DIR / f"{model_name}_pipeline.pkl")
X_test = joblib.load(SAVE_DIR / "X_test.pkl")

# Predict
y_preds = pipe.predict(X_test)
y_probs = pipe.predict_proba(X_test)[:, 1]

# Save output
output = pd.DataFrame({"prediction": y_preds, "probability": y_probs})
output.to_csv(SAVE_DIR / f"{model_name}_predictions.csv", index=False)
print(f"Predictions saved to {SAVE_DIR / f'{model_name}_predictions.csv'}")