GlucoTwin — Model Evaluation Report

1. Evaluation Scope

This report summarizes the results of the current synthetic-data evaluation scripts. The forecast and event evaluation datasets each contain 402 evaluated rows, with no rows skipped for invalid values.

These results are exploratory and do not demonstrate clinical validity or readiness for patient care.

2. Forecast Evaluation

Metric| Model| Persistence Baseline
Mean Absolute Error (MAE)| 8.373 mg/dL| 9.278 mg/dL
Root Mean Squared Error (RMSE)| 10.638 mg/dL| 11.637 mg/dL

The model achieved a 9.75% improvement in MAE and an 8.59% improvement in RMSE relative to the persistence baseline on this evaluation dataset.

3. Event Classification Evaluation

At the evaluated default threshold of 0.5:

- True positives: 14
- True negatives: 351
- False positives: 33
- False negatives: 4
- Accuracy: 0.9080
- Precision: 0.2979
- Recall: 0.7778
- Specificity: 0.9141
- F1 score: 0.4308
- Brier score: 0.0619

Accuracy should be interpreted alongside precision, recall, and the confusion matrix because positive events are relatively uncommon in this dataset.

4. Exploratory Threshold Comparison

The best F1 score among thresholds tested from 0.1 to 0.9 was observed at 0.8.

At threshold 0.8:

- Precision: 0.522
- Recall: 0.667
- F1 score: 0.585
- False positives: 11
- False negatives: 6

Compared with threshold 0.5, threshold 0.8 produced fewer false positives but more false negatives. This is a trade-off, not proof that 0.8 is the appropriate operational threshold.

5. Limitations and Next Steps

- The reported results describe the available evaluation dataset only.
- Synthetic-data performance may not generalize to real-world patient data.
- Threshold selection requires validation on separate data and consideration of the intended use.
- Further work should include independent validation, data-leakage checks, subgroup analysis where appropriate, and calibration assessment.
- No clinical decisions should be based on these exploratory results.

Conclusion

The forecast model outperformed the persistence baseline on the reported error metrics. Event detection showed a recall–precision trade-off, and threshold 0.8 achieved the highest F1 score among the tested thresholds. Independent validation is needed before drawing stronger conclusions.