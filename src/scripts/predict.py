from utils.features import add_features
from src.utils.config import (
    SAVE_PATH,
    label,
    ID_feature
)
import argparse
import sys
from pathlib import Path
import pandas as pd
import joblib

# Paths
ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_PATH))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    args = parser.parse_args()

    model_name = args.model_name
    model_path = SAVE_PATH / f"{model_name}_pipeline.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing saved model: {model_path.name}")
    pipe = joblib.load(model_path)

    raw_test = pd.read_csv(ROOT_PATH / "data" / "test.csv")
    bureau = pd.read_csv(ROOT_PATH / "data" / "bureau.csv")
    bureaubal = pd.read_csv(ROOT_PATH / "data" / "bureau_balance.csv")
    prevapp = pd.read_csv(ROOT_PATH / "data" / "previous_application.csv")
    credit_card = pd.read_csv(ROOT_PATH / "data" / "credit_card_balance.csv")
    installments = pd.read_csv(ROOT_PATH / "data" / "installments_payments.csv")
    pos_cash = pd.read_csv(ROOT_PATH / "data" / "POS_CASH_balance.csv")

    raw_test = add_features(raw_test, bureau, bureaubal,
                            prevapp, credit_card, installments, pos_cash)

    y_preds = pipe.predict_proba(raw_test)[:, 1]

    df = pd.DataFrame({
        ID_feature: raw_test[ID_feature],
        label: y_preds
    })
    out_file = SAVE_PATH / f"{model_name}_submission.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved predictions to {out_file}")
