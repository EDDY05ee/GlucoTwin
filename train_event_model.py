from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

EVENT_THRESHOLD = 180.0  # mg/dL; demonstration definition only

FEATURES = [
    "glucose_mg_dl",
    "glucose_lag_1",
    "glucose_lag_2",
    "glucose_lag_3",
    "heart_rate_bpm",
    "steps_30min",
    "sleep_period_proxy",
    "age_years",
    "diabetes_duration_years",
    "hba1c_percent",
]


def prepare_data():
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

    # Inputs use current and past observations only.
    for lag in (1, 2, 3):
        data[f"glucose_lag_{lag}"] = (
            grouped["glucose_mg_dl"].shift(lag)
        )

    # Labels use future readings and are never model inputs.
    data["future_30min"] = (
        grouped["glucose_mg_dl"].shift(-1)
    )
    data["future_60min"] = (
        grouped["glucose_mg_dl"].shift(-2)
    )

    data["event_within_60min"] = (
        (data["future_30min"] >= EVENT_THRESHOLD)
        | (data["future_60min"] >= EVENT_THRESHOLD)
    ).astype(int)

    data = data.dropna(
        subset=FEATURES + ["future_30min", "future_60min"]
    ).copy()

    return data


def main():
    data = prepare_data()

    # Hold out the latest 20% of observations per patient.
    data["time_rank"] = data.groupby(
        "patient_id"
    )["timestamp"].rank(method="first", pct=True)

    train = data[data["time_rank"] <= 0.80].copy()
    test = data[data["time_rank"] > 0.80].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test split is empty.")

    if train["event_within_60min"].nunique() < 2:
        raise ValueError(
            "Training data contains only one event class. "
            "More varied synthetic data is needed."
        )

    X_train = train[FEATURES]
    y_train = train["event_within_60min"]
    X_test = test[FEATURES]
    y_test = test["event_within_60min"]

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    # Fixed demonstration threshold, not a clinical threshold.
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    accuracy = accuracy_score(y_test, predictions)

    results = pd.DataFrame(
        {
            "patient_id": test["patient_id"].to_numpy(),
            "timestamp": test["timestamp"].to_numpy(),
            "actual_event": y_test.to_numpy(),
            "predicted_event": predictions,
            "predicted_probability": probabilities,
        }
    )

    output_path = DATA_DIR / "event_evaluation.csv"
    results.to_csv(output_path, index=False)

    print("\n=== GlucoTwin Event Model Evaluation ===")
    print("Data: synthetic demonstration data only")
    print("Event: glucose >= 180 mg/dL within 60 minutes")
    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Actual event prevalence in test: {y_test.mean():.1%}")

    print("\nClassification metrics:")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")

    print("\nConfusion matrix:")
    print(f"True negatives:  {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True positives:  {tp}")

    print(f"\nSaved results to: {output_path}")
    print(
        "\nThese metrics are not clinical evidence. "
        "Synthetic data cannot establish real-world safety "
        "or predictive performance."
    )


if __name__ == "__main__":
    main()