from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

HORIZON_STEPS = 1  # 30 minutes, because records are 30 minutes apart.

FEATURES = [
    "glucose_mg_dl",
    "glucose_previous_1",
    "glucose_previous_2",
    "glucose_previous_3",
    "heart_rate_bpm",
    "steps_30min",
    "sleep_period_proxy",
    "age_years",
    "diabetes_duration_years",
    "hba1c_percent",
]


def load_and_prepare_data():
    timeseries = pd.read_csv(
        DATA_DIR / "synthetic_timeseries.csv",
        parse_dates=["timestamp"],
    )

    ehr = pd.read_csv(DATA_DIR / "synthetic_ehr.csv")

    data = timeseries.merge(
        ehr[
            [
                "patient_id",
                "age_years",
                "diabetes_duration_years",
                "hba1c_percent",
            ]
        ],
        on="patient_id",
        how="left",
        validate="many_to_one",
    )

    data = data.sort_values(
        ["patient_id", "timestamp"]
    ).reset_index(drop=True)

    grouped = data.groupby("patient_id", sort=False)

    for lag in (1, 2, 3):
        data[f"glucose_previous_{lag}"] = (
            grouped["glucose_mg_dl"].shift(lag)
        )

    # Future glucose is the target, never a model input.
    data["target_glucose"] = (
        grouped["glucose_mg_dl"].shift(-HORIZON_STEPS)
    )

    data = data.dropna(
        subset=FEATURES + ["target_glucose"]
    ).copy()

    return data


def main():
    data = load_and_prepare_data()

    # Chronological holdout: the latest 20% of timestamps
    # for each patient is reserved for testing.
    data["time_rank"] = data.groupby(
        "patient_id"
    )["timestamp"].rank(method="first", pct=True)

    train = data[data["time_rank"] <= 0.80].copy()
    test = data[data["time_rank"] > 0.80].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test dataset is empty.")

    X_train = train[FEATURES]
    y_train = train["target_glucose"]

    X_test = test[FEATURES]
    y_test = test["target_glucose"]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Persistence baseline: next glucose equals current glucose.
    baseline_predictions = test["glucose_mg_dl"].to_numpy()

    model_mae = mean_absolute_error(y_test, predictions)
    model_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )
    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )

    results = pd.DataFrame(
        {
            "patient_id": test["patient_id"].to_numpy(),
            "timestamp": test["timestamp"].to_numpy(),
            "actual_glucose_mg_dl": y_test.to_numpy(),
            "model_prediction_mg_dl": predictions,
            "persistence_prediction_mg_dl": baseline_predictions,
        }
    )

    results_path = DATA_DIR / "forecast_evaluation.csv"
    results.to_csv(results_path, index=False)

    print("\n=== GlucoTwin Forecast Evaluation ===")
    print("Data: synthetic demonstration records only")
    print("Forecast horizon: 30 minutes")
    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")

    print("\nRandom Forest:")
    print(f"MAE:  {model_mae:.2f} mg/dL")
    print(f"RMSE: {model_rmse:.2f} mg/dL")

    print("\nPersistence baseline:")
    print(f"MAE:  {baseline_mae:.2f} mg/dL")
    print(f"RMSE: {baseline_rmse:.2f} mg/dL")

    print(f"\nEvaluation predictions saved to: {results_path}")
    print(
        "\nThese metrics describe a synthetic-data experiment. "
        "They do not establish clinical accuracy."
    )


if __name__ == "__main__":
    main()