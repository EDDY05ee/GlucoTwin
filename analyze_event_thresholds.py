from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "data" / "event_evaluation.csv"


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main():
    print("GlucoTwin — Event Threshold Analysis")
    print("Read-only analysis; no files will be changed.")

    if not CSV_PATH.is_file():
        print(f"ERROR: File not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    required = ["actual_event", "predicted_probability"]

    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"ERROR: Required columns missing: {missing}")
        return

    data = df[required].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        print("ERROR: No valid rows available.")
        return

    actual = data["actual_event"].to_numpy()
    probability = data["predicted_probability"].to_numpy()

    if not np.isin(actual, [0, 1]).all():
        print("ERROR: actual_event must contain only 0 and 1.")
        return

    if ((probability < 0) | (probability > 1)).any():
        print("ERROR: Probabilities must be between 0 and 1.")
        return

    actual = actual.astype(int)

    print(f"\nRows evaluated: {len(data):,}")
    print("\nTHRESHOLD COMPARISON")
    print(
        f"{'Threshold':>9} {'Precision':>10} {'Recall':>9} "
        f"{'F1':>8} {'FP':>6} {'FN':>6}"
    )

    results = []

    for threshold in np.arange(0.1, 1.0, 0.1):
        predicted = (probability >= threshold).astype(int)

        tp = int(((actual == 1) & (predicted == 1)).sum())
        tn = int(((actual == 0) & (predicted == 0)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "tn": tn,
            }
        )

        print(
            f"{threshold:9.1f} {precision:10.3f} {recall:9.3f} "
            f"{f1:8.3f} {fp:6d} {fn:6d}"
        )

    best = max(results, key=lambda row: row["f1"])
    print("\nBest F1 among tested thresholds:")
    print(f"Threshold: {best['threshold']:.1f}")
    print(f"Precision: {best['precision']:.3f}")
    print(f"Recall:    {best['recall']:.3f}")
    print(f"F1 score:  {best['f1']:.3f}")
    print(f"False positives: {best['fp']}")
    print(f"False negatives: {best['fn']}")

    print("\nThis is exploratory analysis, not clinical threshold selection.")
    print("Validate any proposed threshold on separate data before adoption.")


if __name__ == "__main__":
    main()