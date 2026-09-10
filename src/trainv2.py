import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import joblib


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_anomalies_2020_2024.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "day6"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_FILE = (
    MODEL_DIR
    / "isolation_forest_v2.joblib"
)

FEATURE_FILE = (
    MODEL_DIR
    / "feature_columns_v2.joblib"
)

TEST_RESULT_FILE = (
    REPORT_DIR
    / "day6_v2_test_results.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "day6_v2_evaluation_summary.csv"
)

TYPE_RESULT_FILE = (
    REPORT_DIR
    / "day6_v2_anomaly_type_results.csv"
)


# ============================================================
# 2. PRINT PROJECT INFORMATION
# ============================================================

print("=" * 70)
print("VAYU PRAHARI - ISOLATION FOREST V2")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nDataset:")
print(DATA_FILE)

print("\nModel output:")
print(MODEL_FILE)


# ============================================================
# 3. CHECK DATASET
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}"
    )


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\n")
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nShape:")
print(df.shape)


# ============================================================
# 5. CONVERT TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

if df["timestamp"].isna().any():

    raise ValueError(
        "Timestamp conversion produced missing values."
    )


# ============================================================
# 6. DEFINE EXACT 24 MODEL FEATURES
# ============================================================

feature_columns = [
    "temperature",
    "humidity",
    "pressure",

    "temp_prev",
    "humidity_prev",
    "pressure_prev",

    "temp_diff",
    "humidity_diff",
    "pressure_diff",

    "temp_rate",
    "humidity_rate",
    "pressure_rate",

    "temp_roll_mean_3",
    "humidity_roll_mean_3",
    "pressure_roll_mean_3",

    "temp_roll_std_3",
    "humidity_roll_std_3",
    "pressure_roll_std_3",

    "hour",
    "month",

    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos"
]


print("\n")
print("=" * 70)
print("MODEL FEATURES")
print("=" * 70)

print("\nNumber of features:")
print(len(feature_columns))

print("\nFeatures:")

for i, feature in enumerate(
    feature_columns,
    start=1
):

    print(
        f"{i:2d}. {feature}"
    )


# ============================================================
# 7. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    feature_columns
    + [
        "anomaly",
        "anomaly_type"
    ]
)

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# 8. CONVERT FEATURES TO NUMERIC
# ============================================================

for column in feature_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 9. CREATE YEAR COLUMN
# ============================================================

df["year"] = (
    df["timestamp"].dt.year
)


# ============================================================
# 10. SHOW OVERALL ANOMALY COUNTS
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL DATASET")
print("=" * 70)

print("\nNormal / anomaly counts:")

print(
    df["anomaly"].value_counts()
)


print("\nAnomaly type counts:")

print(
    df["anomaly_type"].value_counts()
)


# ============================================================
# 11. TRAINING DATA
# ============================================================

print("\n")
print("=" * 70)
print("PREPARING TRAINING DATA")
print("=" * 70)

train_df = df[
    (df["year"] <= 2023)
    &
    (df["anomaly"] == 0)
].copy()

print("\nTraining rows before dropping missing features:")
print(len(train_df))


# ============================================================
# 12. DROP MISSING TRAINING FEATURES
# ============================================================

train_df = train_df.dropna(
    subset=feature_columns
).copy()

print(
    "\nTraining rows after dropping missing features:"
)

print(
    len(train_df)
)


# ============================================================
# 13. CREATE X_train
# ============================================================

X_train = train_df[
    feature_columns
].copy()


print("\nX_train shape:")
print(X_train.shape)


# ============================================================
# 14. TRAIN ISOLATION FOREST V2
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING ISOLATION FOREST V2")
print("=" * 70)

model = IsolationForest(
    n_estimators=300,
    max_samples=0.8,
    max_features=1.0,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)


print("\nModel configuration:")

print("n_estimators   = 300")
print("max_samples    = 0.8")
print("max_features   = 1.0")
print("contamination  = 0.01")
print("random_state   = 42")


model.fit(X_train)


print("\nTraining completed.")


# ============================================================
# 15. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    feature_columns,
    FEATURE_FILE
)


print("\nModel saved:")
print(MODEL_FILE)

print("\nFeature list saved:")
print(FEATURE_FILE)


# ============================================================
# 16. PREPARE 2024 TEST DATA
# ============================================================

print("\n")
print("=" * 70)
print("PREPARING 2024 TEST DATA")
print("=" * 70)

test_df = df[
    df["year"] == 2024
].copy()

print("\n2024 rows before dropping missing features:")
print(len(test_df))


# ============================================================
# 17. DROP MISSING TEST FEATURES
# ============================================================

test_df = test_df.dropna(
    subset=feature_columns
).copy()

print(
    "\n2024 rows after dropping missing features:"
)

print(
    len(test_df)
)


# ============================================================
# 18. CREATE TEST MATRICES
# ============================================================

X_test = test_df[
    feature_columns
].copy()

y_test = test_df[
    "anomaly"
].astype(int)


print("\nX_test shape:")
print(X_test.shape)

print("\ny_test shape:")
print(y_test.shape)


print("\nActual test labels:")

print(
    y_test.value_counts()
)


# ============================================================
# 19. PREDICTIONS
# ============================================================

print("\n")
print("=" * 70)
print("RUNNING PREDICTIONS")
print("=" * 70)

raw_predictions = model.predict(
    X_test
)


# Isolation Forest:
# +1 = normal
# -1 = anomaly

predictions = (
    raw_predictions == -1
).astype(int)


# ============================================================
# 20. ANOMALY SCORES
# ============================================================

anomaly_scores = model.decision_function(
    X_test
)


# ============================================================
# 21. STORE RESULTS
# ============================================================

test_df["prediction"] = predictions

test_df["anomaly_score"] = anomaly_scores


# ============================================================
# 22. BASIC COUNTS
# ============================================================

actual_anomalies = int(
    y_test.sum()
)

predicted_anomalies = int(
    predictions.sum()
)

actual_normal = int(
    (y_test == 0).sum()
)

predicted_normal = int(
    (predictions == 0).sum()
)


print("\nActual anomalies:")
print(actual_anomalies)

print("\nPredicted anomalies:")
print(predicted_anomalies)

print("\nActual normal:")
print(actual_normal)

print("\nPredicted normal:")
print(predicted_normal)


# ============================================================
# 23. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

tn, fp, fn, tp = cm.ravel()


print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

print("\nTrue Negatives:")
print(tn)

print("\nFalse Positives:")
print(fp)

print("\nFalse Negatives:")
print(fn)

print("\nTrue Positives:")
print(tp)


# ============================================================
# 24. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

fpr = fp / (
    fp + tn
) if (
    fp + tn
) > 0 else 0


# ============================================================
# 25. PRINT METRICS
# ============================================================

print("\n")
print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(
    f"\nAccuracy : {accuracy:.6f}"
)

print(
    f"Precision: {precision:.6f}"
)

print(
    f"Recall   : {recall:.6f}"
)

print(
    f"F1 Score : {f1:.6f}"
)

print(
    f"FPR      : {fpr:.6f}"
)


# ============================================================
# 26. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Normal",
            "Anomaly"
        ],
        zero_division=0
    )
)


# ============================================================
# 27. ACTUAL ANOMALY DETECTION BY TYPE
# ============================================================

print("\n")
print("=" * 70)
print("ANOMALY TYPE DETECTION")
print("=" * 70)

actual_anomaly_df = test_df[
    test_df["anomaly"] == 1
].copy()


type_results = []

for anomaly_type in sorted(
    actual_anomaly_df[
        "anomaly_type"
    ].unique()
):

    type_df = actual_anomaly_df[
        actual_anomaly_df[
            "anomaly_type"
        ] == anomaly_type
    ]

    total = len(type_df)

    detected = int(
        type_df["prediction"].sum()
    )

    missed = total - detected

    detection_rate = (
        detected / total
        if total > 0
        else 0
    )

    type_results.append({
        "anomaly_type": anomaly_type,
        "actual_count": total,
        "detected_count": detected,
        "missed_count": missed,
        "detection_rate": detection_rate
    })


type_results_df = pd.DataFrame(
    type_results
)


print(
    type_results_df.to_string(
        index=False
    )
)


# ============================================================
# 28. SHOW DETECTED REAL ANOMALIES
# ============================================================

print("\n")
print("=" * 70)
print("ACTUAL ANOMALIES DETECTED BY MODEL")
print("=" * 70)

detected_actual = test_df[
    (
        test_df["anomaly"] == 1
    )
    &
    (
        test_df["prediction"] == 1
    )
][
    [
        "timestamp",
        "temperature",
        "humidity",
        "pressure",
        "anomaly_type",
        "anomaly_score",
        "prediction"
    ]
]


if len(detected_actual) > 0:

    print(
        detected_actual.to_string(
            index=False
        )
    )

else:

    print(
        "No actual anomalies detected."
    )


# ============================================================
# 29. SHOW MISSED REAL ANOMALIES
# ============================================================

print("\n")
print("=" * 70)
print("ACTUAL ANOMALIES MISSED BY MODEL")
print("=" * 70)

missed_actual = test_df[
    (
        test_df["anomaly"] == 1
    )
    &
    (
        test_df["prediction"] == 0
    )
][
    [
        "timestamp",
        "temperature",
        "humidity",
        "pressure",
        "anomaly_type",
        "anomaly_score",
        "prediction"
    ]
]


if len(missed_actual) > 0:

    print(
        missed_actual.to_string(
            index=False
        )
    )

else:

    print(
        "No actual anomalies were missed."
    )


# ============================================================
# 30. SAVE TEST RESULTS
# ============================================================

test_df.to_csv(
    TEST_RESULT_FILE,
    index=False
)


# ============================================================
# 31. SAVE EVALUATION SUMMARY
# ============================================================

summary_df = pd.DataFrame([
    {
        "model": "Isolation Forest v2",
        "n_estimators": 300,
        "max_samples": 0.8,
        "max_features": 1.0,
        "contamination": 0.01,

        "training_rows": len(X_train),
        "test_rows": len(X_test),

        "actual_anomalies": actual_anomalies,
        "predicted_anomalies": predicted_anomalies,

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,

        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    }
])


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 32. SAVE ANOMALY TYPE RESULTS
# ============================================================

type_results_df.to_csv(
    TYPE_RESULT_FILE,
    index=False
)


# ============================================================
# 33. FINAL FILE INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print("\nModel:")
print(MODEL_FILE)

print("\nFeature list:")
print(FEATURE_FILE)

print("\nTest results:")
print(TEST_RESULT_FILE)

print("\nEvaluation summary:")
print(SUMMARY_FILE)

print("\nAnomaly type results:")
print(TYPE_RESULT_FILE)


# ============================================================
# 34. SUCCESS
# ============================================================

print("\n")
print("=" * 70)
print("ISOLATION FOREST V2 COMPLETE")
print("=" * 70)

print("\nModel successfully trained and saved.")