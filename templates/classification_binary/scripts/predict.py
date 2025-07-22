import argparse
import sys
from pathlib import Path
import pandas as pd
import joblib
import yaml

# Paths
ROOT = Path(__file__).resolve().parent.parent
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


def load_test_data(config):
    df = joblib.load(SAVE_DIR / "X_test.pkl")

    drop_features = config["preprocessing"].get("drop_features", [])
    df = df.drop(columns=[col for col in drop_features if col in df.columns], errors="ignore")

    return df


def predict_with_pipeline(pipeline, X):
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline.named_steps['model'], 'predict_proba') else None
    return preds, probs


def map_predictions_to_labels(preds, config):
    label = config["general"]["label"]
    value_map = config["preprocessing"].get("value_mappings", {}).get(label, {})
    inverse_map = {v: k for k, v in value_map.items()}
    return pd.Series(preds).map(inverse_map)


def save_submission(ids, preds, model_name):
    df = pd.DataFrame({
        "id": ids,
        "Personality": preds
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

    if "id" not in raw_test.columns:
        raise ValueError("Test file must include an 'id' column")

    raw_preds, _ = predict_with_pipeline(pipe, X_test)
    readable_preds = map_predictions_to_labels(raw_preds, config)

    save_submission(raw_test["id"], readable_preds, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions using saved pipeline")
    parser.add_argument("model_name", type=str, help="Model name to use (must match saved .pkl file)")
    args = parser.parse_args()
    main(args.model_name)