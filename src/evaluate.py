from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "day5_vscode_test_results.csv"
)


# ============================================================
# 2. CHECK RESULT FILE
# ============================================================

if not RESULT_PATH.exists():
    raise FileNotFoundError(
        "Prediction results not found.\n"
        "Run: python src/predict.py"
    )


# ============================================================
# 3. LOAD RESULTS
# ============================================================

df = pd.read_csv(
    RESULT_PATH
)

print("=" * 60)
print("VAYU PRAHARI - MODEL EVALUATION")
print("=" * 60)


# ============================================================
# 4. ACTUAL / PREDICTED LABELS
# ============================================================

y_true = df["anomaly"]

y_pred = df["prediction"]


# ============================================================
# 5. METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ============================================================
# 6. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Normal",
            "Anomaly"
        ],
        zero_division=0
    )
)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


tn, fp, fn, tp = cm.ravel()


print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


# ============================================================
# 8. FALSE POSITIVE RATE
# ============================================================

false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0
)

print(
    "\nFalse Positive Rate:",
    false_positive_rate
)


# ============================================================
# 9. CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "Normal",
        "Anomaly"
    ],
    yticklabels=[
        "Normal",
        "Anomaly"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Vayu Prahari - Isolation Forest Confusion Matrix"
)

plt.tight_layout()

plt.show()


# ============================================================
# 10. DETECTED ANOMALIES
# ============================================================

detected = df[
    df["prediction"] == 1
]

print(
    "\nDetected anomalies:",
    len(detected)
)


# ============================================================
# 11. ACTUAL ANOMALIES
# ============================================================

actual = df[
    df["anomaly"] == 1
]

print(
    "Actual anomalies:",
    len(actual)
)


# ============================================================
# 12. MISSED ANOMALIES
# ============================================================

missed = df[
    (df["anomaly"] == 1)
    &
    (df["prediction"] == 0)
]

print(
    "Missed anomalies:",
    len(missed)
)


# ============================================================
# 13. FALSE ALARMS
# ============================================================

false_alarms = df[
    (df["anomaly"] == 0)
    &
    (df["prediction"] == 1)
]

print(
    "False alarms:",
    len(false_alarms)
)


# ============================================================
# 14. PER-ANOMALY-TYPE ANALYSIS
# ============================================================

type_results = []

for anomaly_type, group in df.groupby(
    "anomaly_type"
):

    if anomaly_type == "normal":
        continue

    total = len(group)

    detected_count = (
        group["prediction"] == 1
    ).sum()

    detection_rate = (
        detected_count / total
        if total > 0
        else 0
    )

    type_results.append({
        "anomaly_type": anomaly_type,
        "samples": total,
        "detected": detected_count,
        "detection_rate": detection_rate
    })


type_results_df = pd.DataFrame(
    type_results
)


print("\nPer-anomaly-type detection:")

print(
    type_results_df.to_string(index=False)
)