import argparse
import sys
from pathlib import Path
import pandas as pd
import joblib
import yaml

# Paths
ROOT = Path(__file__).resolve().parents[2]
SAVE_DIR = ROOT / "saved"
CONFIG_PATH = ROOT / "config.yaml"
sys.path.append(str(ROOT))



def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)



def load_model(model_name):
    model_path = SAVE_DIR / f"{model_name}_pipeline.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing saved model: {model_path.name}")
    return joblib.load(model_path)



def predict_with_pipeline(pipe, X):
    config = load_config()
    threshold = config["general"]["threshold"]

    if hasattr(pipe.named_steps['model'], "predict_proba"):
        y_probs = pipe.predict_proba(X)[:, 1]
        y_preds = (y_probs >= threshold).astype(int)
    else:
        y_probs = None
        y_preds = pipe.predict(X)

    return y_preds, y_probs



def save_submission(ids, preds, model_name, label):
    df = pd.DataFrame({
        "id": ids,
        label: preds
    })
    out_file = SAVE_DIR / f"{model_name}_submission.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved predictions to {out_file}")



def main(model_name):
    config = load_config()
    pipe = load_model(model_name)

    # Load test data and raw IDs
    X_test = joblib.load(SAVE_DIR / "X_test.pkl")
    raw_test = pd.read_csv(ROOT / "data" / "test.csv")

    cleaning_pipeline = joblib.load(SAVE_DIR / "cleaning_pipeline.pkl")
    preprocessor = joblib.load(SAVE_DIR / "preprocessor.pkl")

    X_test = cleaning_pipeline.transform(raw_test)
    X_test = preprocessor.transform(X_test)

    y_preds = predict_with_pipeline(pipe, X_test)

    label = config["general"]["label"]
    save_submission(raw_test["id"], y_preds, model_name, label)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions using saved pipeline")
    parser.add_argument("model_name", type=str, help="Model name to use (must match saved .pkl file)")
    args = parser.parse_args()
    main(args.model_name)