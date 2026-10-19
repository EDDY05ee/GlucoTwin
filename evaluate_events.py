from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "data" / "event_evaluation.csv"


def main():
    print("GlucoTwin — Event Evaluation")
    print("Read-only report; no data will be changed.")

    if not CSV_PATH.is_file():
        print(f"ERROR: File not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    required = ["actual_event", "predicted_event", "predicted_probability"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"ERROR: Required columns missing: {missing}")
        return

    values = df[required].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()

    if values.empty:
        print("ERROR: No valid evaluation rows.")
        return

    actual = values["actual_event"].to_numpy()
    predicted = values["predicted_event"].to_numpy()
    probability = values["predicted_probability"].to_numpy()

    if not np.isin(actual, [0, 1]).all():
        print("ERROR: actual_event must use binary labels 0 and 1.")
        return

    if not np.isin(predicted, [0, 1]).all():
        print("ERROR: predicted_event must use binary labels 0 and 1.")
        return

    if ((probability < 0) | (probability > 1)).any():
        print("ERROR: predicted_probability must be between 0 and 1.")
        return

    actual = actual.astype(int)
    predicted = predicted.astype(int)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else float("nan")

    accuracy = ratio(tp + tn, len(actual))
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    brier = float(np.mean((probability - actual) ** 2))

    print(f"\nSource rows: {len(df):,}")
    print(f"Rows evaluated: {len(values):,}")
    print(f"Rows skipped for invalid values: {len(df) - len(values):,}")

    print("\nCONFUSION MATRIX")
    print(f"True positives:  {tp}")
    print(f"True negatives:  {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")

    print("\nCLASSIFICATION METRICS")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 score:    {f1:.4f}")

    print("\nPROBABILITY METRIC")
    print(f"Brier score: {brier:.4f} (lower is better)")

    print("\nMetrics describe this dataset, not clinical readiness.")


if __name__ == "__main__":
    main()