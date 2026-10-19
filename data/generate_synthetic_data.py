from pathlib import Path

import numpy as np
import pandas as pd


# All records generated here are synthetic demonstration data.
OUTPUT_DIR = Path(__file__).resolve().parent
RNG = np.random.default_rng(2026)

PATIENTS = [
    {
        "patient_id": "GT-001",
        "age_years": 52,
        "diabetes_duration_years": 8,
        "hba1c_percent": 7.4,
        "comorbidity_count": 2,
    },
    {
        "patient_id": "GT-002",
        "age_years": 46,
        "diabetes_duration_years": 5,
        "hba1c_percent": 6.8,
        "comorbidity_count": 1,
    },
    {
        "patient_id": "GT-003",
        "age_years": 61,
        "diabetes_duration_years": 12,
        "hba1c_percent": 8.1,
        "comorbidity_count": 3,
    },
]


def generate_ehr():
    """Create static synthetic patient profiles."""
    records = []

    for patient in PATIENTS:
        record = patient.copy()
        record["data_type"] = "SYNTHETIC_DEMO"
        records.append(record)

    return pd.DataFrame(records)


def generate_timeseries(days=14):
    """Create synthetic glucose and wearable observations."""
    records = []
    periods_per_day = 48  # One observation every 30 minutes.

    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("30min"),
        periods=days * periods_per_day,
        freq="30min",
    )

    baseline_values = {
        "GT-001": 132,
        "GT-002": 118,
        "GT-003": 145,
    }

    for patient in PATIENTS:
        patient_id = patient["patient_id"]
        baseline = baseline_values[patient_id]

        patient_rng = np.random.default_rng(
            2026 + int(patient_id[-3:])
        )

        for timestamp in timestamps:
            hour = timestamp.hour + timestamp.minute / 60

            # Illustrative daily glucose pattern.
            daily_pattern = 15 * np.sin(
                2 * np.pi * (hour - 6) / 24
            )

            # Synthetic meal-related glucose variation.
            meal_effect = sum(
                amplitude
                * np.exp(-0.5 * ((hour - meal_hour) / 0.8) ** 2)
                for meal_hour, amplitude in [
                    (8, 22),
                    (13, 28),
                    (19, 25),
                ]
            )

            glucose = (
                baseline
                + daily_pattern
                + meal_effect
                + patient_rng.normal(0, 8)
            )

            # Synthetic wearable-like signals.
            is_sleep_period = hour < 7 or hour >= 23

            heart_rate = (
                62 if is_sleep_period else 76
            ) + patient_rng.normal(0, 5)

            steps = max(
                0,
                int(
                    patient_rng.gamma(2, 18)
                    if 7 <= hour < 22
                    else patient_rng.gamma(1, 2)
                ),
            )

            records.append(
                {
                    "patient_id": patient_id,
                    "timestamp": timestamp,
                    "glucose_mg_dl": round(
                        float(np.clip(glucose, 65, 280)), 1
                    ),
                    "heart_rate_bpm": round(
                        float(np.clip(heart_rate, 40, 150)), 1
                    ),
                    "steps_30min": steps,
                    "sleep_period_proxy": int(is_sleep_period),
                    "data_type": "SYNTHETIC_DEMO",
                }
            )

    return pd.DataFrame(records)


def main():
    ehr = generate_ehr()
    timeseries = generate_timeseries()

    ehr_path = OUTPUT_DIR / "synthetic_ehr.csv"
    timeseries_path = OUTPUT_DIR / "synthetic_timeseries.csv"

    ehr.to_csv(ehr_path, index=False)
    timeseries.to_csv(timeseries_path, index=False)

    print("GlucoTwin synthetic data generated successfully.")
    print(f"EHR records: {len(ehr)}")
    print(f"Time-series observations: {len(timeseries)}")
    print(f"Patients: {timeseries['patient_id'].nunique()}")
    print(f"EHR file: {ehr_path}")
    print(f"Time-series file: {timeseries_path}")
    print("All generated records are synthetic demonstration data.")


if __name__ == "__main__":
    main()