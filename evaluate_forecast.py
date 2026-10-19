from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "data" / "forecast_evaluation.csv"


def calculate_metrics(actual, predicted):
    errors = predicted - actual
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    return float(mae), float(rmse)


def main():
    print("GlucoTwin — Forecast Evaluation")
    print("Read-only report; no data will be changed.")

    if not CSV_PATH.is_file():
        print(f"ERROR: File not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    required = [
        "actual_glucose_mg_dl",
        "model_prediction_mg_dl",
        "persistence_prediction_mg_dl",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        print(f"ERROR: Required columns missing: {missing}")
        return

    values = df[required].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()

    if values.empty:
        print("ERROR: No valid rows available for evaluation.")
        return

    actual = values["actual_glucose_mg_dl"]
    model = values["model_prediction_mg_dl"]
    baseline = values["persistence_prediction_mg_dl"]

    model_mae, model_rmse = calculate_metrics(actual, model)
    base_mae, base_rmse = calculate_metrics(actual, baseline)

    print(f"\nSource rows: {len(df):,}")
    print(f"Rows evaluated: {len(values):,}")
    print(f"Rows skipped for invalid values: {len(df) - len(values):,}")

    print("\nMODEL")
    print(f"MAE:  {model_mae:.3f} mg/dL")
    print(f"RMSE: {model_rmse:.3f} mg/dL")

    print("\nPERSISTENCE BASELINE")
    print(f"MAE:  {base_mae:.3f} mg/dL")
    print(f"RMSE: {base_rmse:.3f} mg/dL")

    print("\nCOMPARISON")
    if base_mae > 0:
        improvement = (base_mae - model_mae) / base_mae * 100
        print(f"MAE improvement: {improvement:+.2f}%")
    else:
        print("MAE improvement: unavailable (baseline MAE is zero)")

    if base_rmse > 0:
        improvement = (base_rmse - model_rmse) / base_rmse * 100
        print(f"RMSE improvement: {improvement:+.2f}%")
    else:
        print("RMSE improvement: unavailable (baseline RMSE is zero)")

    print("\nLower MAE and RMSE indicate smaller prediction errors.")
    print("Results describe this evaluation dataset, not clinical readiness.")


if __name__ == "__main__":
    main()