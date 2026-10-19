from pathlib import Path

import pandas as pd

# Read-only audit: this script never writes to or changes your data.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

DATASETS = {
    "Synthetic EHR": "synthetic_ehr.csv",
    "Synthetic time series": "synthetic_timeseries.csv",
    "Forecast evaluation": "forecast_evaluation.csv",
    "Event evaluation (optional)": "event_evaluation.csv",
}


def audit_csv(label: str, filename: str) -> None:
    path = DATA_DIR / filename

    print(f"\n{'=' * 60}")
    print(f"Dataset: {label}")
    print(f"File:    data/{filename}")

    if not path.is_file():
        print("STATUS:  NOT FOUND")
        return

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"STATUS:  READ ERROR — {exc}")
        return

    print("STATUS:  FOUND")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    if df.empty:
        print("WARNING: This CSV contains no data rows.")
        return

    print(f"Duplicate rows: {df.duplicated().sum():,}")

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("Missing values: None detected")
    else:
        print("Missing values by column:")
        for column, count in missing.items():
            print(f"  - {column}: {count:,}")

    print("Column names:")
    for column in df.columns:
        print(f"  - {column}")


def main() -> None:
    print("GlucoTwin — Read-only data audit")
    print(f"Project: {PROJECT_DIR}")

    if not DATA_DIR.is_dir():
        print(f"\nERROR: Data folder not found: {DATA_DIR}")
        return

    for label, filename in DATASETS.items():
        audit_csv(label, filename)

    print("\n" + "=" * 60)
    print("Audit finished. No files were modified.")
    print("Review warnings before making any data changes.")


if __name__ == "__main__":
    main()